// realesrgan implemented with ncnn library
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
#include <codecvt>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <net.h> //include the ncnn header file
namespace fs = std::filesystem;


#if _WIN32
// image decoder and encoder with wic
#include "wic_image.h"
#else // _WIN32
// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // _WIN32
#include "webp_image.h"

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "realesrgan.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: realesrgan-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -s scale             upscale ratio (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -m model-path        folder path to the pre-trained models. default=models\n");
    fprintf(stderr, "  -n model-name        model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x                   enable tta mode\n");
    fprintf(stderr, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  -v                   verbose output\n");
}

#include <condition_variable>
#include <atomic>

class NoResetEvent
{
public:
    NoResetEvent() : _state(false) {}
    NoResetEvent(const NoResetEvent& other) = delete;
    

    void WaitOne() {
        std::unique_lock<std::mutex> lock(sync);
        while (!_state) {
            underlying.wait(lock);
        }
    }

    void Set() {
        std::unique_lock<std::mutex> lock(sync);
        _state = true;
        underlying.notify_all();
    }

private:
    std::condition_variable underlying;
    std::mutex sync;
    std::atomic<bool>  _state;
};

class Task
{
public:
    int id;

    cv::Mat originalFrame;
    ncnn::Mat outimage;

    std::shared_ptr<NoResetEvent> e;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 100) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int scale;
    
    cv::VideoCapture cap;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    cv::VideoCapture cap = ltp->cap;
    int scale = ltp->scale;
    
    int i = 0;
    cv::Mat cvImage;
    while (cap.read(cvImage)) {
        Task v;
        v.e = std::make_shared<NoResetEvent>();
        v.id = i;
        v.originalFrame = cvImage.clone();

        
        int w = cvImage.cols;
        int h = cvImage.rows;
        int c = cvImage.channels();
        
        v.outimage = ncnn::Mat(w * scale, h * scale, (size_t)c, c);
        
        i++;
        
        toproc.put(v);
        tosave.put(v);
    }

    return 0;
}

class ProcThreadParams
{
public:
    const RealESRGAN* realesrgan;
    int scale;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RealESRGAN* realesrgan = ptp->realesrgan;
    
    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;
        
        auto cvImage = v.originalFrame;
        
        int w = cvImage.cols;
        int h = cvImage.rows;
        int c = cvImage.channels();
        
        auto pixeldata = cvImage.data;
        cv::Mat img(h, w, (c == 3 ? CV_8UC3 : CV_8UC4), pixeldata);
        
        auto ncnn_img = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
            
        realesrgan->process(ncnn_img, v.outimage);
        v.e->Set();
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
    cv::VideoWriter writer;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;

    auto writer = stp->writer;

    auto start = std::chrono::high_resolution_clock::now();
    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        v.e->WaitOne();

        auto processed_ncnn_img = v.outimage;
        auto c = v.originalFrame.channels();
        auto newImage = cv::Mat(processed_ncnn_img.h, processed_ncnn_img.w, (c == 3 ? CV_8UC3 : CV_8UC4), processed_ncnn_img.data);

        writer.write(newImage);
        auto finish = std::chrono::high_resolution_clock::now();
        
        if (v.id % 5000 == 0)
        {
            std::chrono::duration<double> duration = finish - start;
            std::cout << v.id  << " frames processed " << duration.count() << " seconds" << std::endl;
        }
        
    }
    
    return 0;
}


#if _WIN32

int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    path_t inputpath;
    path_t outputpath;
    int scale = 3;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models");
    path_t modelname = PATHSTR("realesr-animevideov3");
    std::vector<int> gpuid;
    std::vector<int> jobs_proc;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:s:t:m:n:g:j:f:vxh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L's':
            scale = _wtoi(optarg);
            break;
        case L't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'n':
            modelname = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'f':
            format = optarg;
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:m:n:g:j:f:vxh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 's':
            scale = atoi(optarg);
            break;
        case 't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'n':
            modelname = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    if (inputpath.empty() || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i=0; i<(int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }


    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    if (format != PATHSTR("png") && format != PATHSTR("webp") && format != PATHSTR("jpg"))
    {
        fprintf(stderr, "invalid format argument\n");
        return -1;
    }

    // collect input and output filepath
    {
        if (!path_is_directory(inputpath) && !path_is_directory(outputpath))
        {
            // ok
        }
        else
        {
            fprintf(stderr, "inputpath and outputpath must be either file or directory at the same time\n");
            return -1;
        }
    }

    int prepadding = 0;

    if (model.find(PATHSTR("models")) != path_t::npos
        || model.find(PATHSTR("models2")) != path_t::npos)
    {
        prepadding = 10;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    // if (modelname.find(PATHSTR("realesrgan-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("realesrnet-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("esrgan-x4")) != path_t::npos)
    // {}
    // else
    // {
    //     fprintf(stderr, "unknown model name\n");
    //     return -1;
    // }

#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        swprintf(parampath, 256, L"%s/%s-x%d.param", model.c_str(), modelname.c_str(), scale);
        swprintf(modelpath, 256, L"%s/%s-x%d.bin", model.c_str(), modelname.c_str(), scale);
    }
    else{
        swprintf(parampath, 256, L"%s/%s.param", model.c_str(), modelname.c_str());
        swprintf(modelpath, 256, L"%s/%s.bin", model.c_str(), modelname.c_str());
    }

#else
    char parampath[256];
    char modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        sprintf(parampath, "%s/%s-x%s.param", model.c_str(), modelname.c_str(), std::to_string(scale).c_str());
        sprintf(modelpath, "%s/%s-x%s.bin", model.c_str(), modelname.c_str(), std::to_string(scale).c_str());
    }
    else{
        sprintf(parampath, "%s/%s.param", model.c_str(), modelname.c_str());
        sprintf(modelpath, "%s/%s.bin", model.c_str(), modelname.c_str());
    }
#endif

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

#if _WIN32
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
#endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < 0 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        int gpu_queue_count = ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    for (int i=0; i<use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 300;
            else if (heap_budget > 550)
                tilesize[i] = 100;
            else if (heap_budget > 190)
                tilesize[i] = 64;
            else
                tilesize[i] = 32;
        }
    }

    {
        std::vector<RealESRGAN*> realesrgan(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);

            realesrgan[i]->load(paramfullpath, modelfullpath);

            realesrgan[i]->scale = scale;
            realesrgan[i]->tilesize = tilesize[i];
            realesrgan[i]->prepadding = prepadding;
        }


        // return 0;
        // main routine
        {
            // load image
            LoadThreadParams ltp;
            ltp.scale = scale;
            
            std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
            std::string utf8_inputpath = converter.to_bytes(inputpath);
            cv::VideoCapture cap(utf8_inputpath);
            ltp.cap = cap;
            
            ncnn::Thread load_thread(load, (void*)&ltp);

            // realesrgan proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].realesrgan = realesrgan[i];
                ptp[i].scale = scale;
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    for (int j=0; j<jobs_proc[i]; j++)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            
            auto fps = cap.get(cv::CAP_PROP_FPS);
            auto codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            auto frame_size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH) * scale, cap.get(cv::CAP_PROP_FRAME_HEIGHT) * scale);

            std::string utf8_outputpath = converter.to_bytes(outputpath);
            
            cv::VideoWriter writer(utf8_outputpath, codec, fps, frame_size);

            stp.writer = writer;

            
            auto start = std::chrono::high_resolution_clock::now();
            
            ncnn::Thread* saveThread = new ncnn::Thread(save, (void*)&stp);

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

             tosave.put(end);

             saveThread->join();
             delete saveThread;

            cap.release();
            writer.release();
            
            auto finish = std::chrono::high_resolution_clock::now();

            // Calculate the duration
            std::chrono::duration<double> duration = finish - start;

            // Print the duration to stdout
            std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete realesrgan[i];
        }
        
        realesrgan.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
