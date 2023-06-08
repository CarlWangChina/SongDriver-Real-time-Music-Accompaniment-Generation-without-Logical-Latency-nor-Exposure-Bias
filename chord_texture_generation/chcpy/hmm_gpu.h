#pragma once
#include "gpu.h"
namespace chcpy {
namespace gpu {

//gpu端
constexpr char hmm_shader[] = R"(

#version 320 es

uniform int M;
uniform int N;
uniform int seq_i;//seq.at(i)

layout(local_size_x = 1) in;

layout(binding = 0) readonly buffer  Input0{
    float data[];
} A_log;//状态转移矩阵A[vid][vid]，格式[N,N]

layout(binding = 1) readonly buffer  Input1{
    float data[];
} B_log;//发射矩阵，B[vid][kid]，格式[N,M]

layout(binding = 2) readonly buffer Input2 {
    float data[];
} dp_last;//上一次的值

layout(binding = 3) writeonly buffer Output0 {
    float data[];
} output0;

void main(){
    const float infinity = 1. / 0.;//定义无限大常量
    int idx = int(gl_GlobalInvocationID.x);//对应cpu版的j
    float dpi_val = -infinity;
    int ptr_val = 0;
    float base = B_log.data[idx*M + seq_i];
    for (int k = 0; k < N; ++k) {
        float score = base + dp_last.data[k] + A_log.data[k*N + idx];
        if (!isinf(score) && score > dpi_val) {
            dpi_val = score;
            ptr_val = k;
        }
    }
    output0.data[idx*2] = dpi_val;
    output0.data[idx*2+1] = float(ptr_val);
}

)";

class hmm_t {
   private:
    float* A_log;
    GLuint A_log_gpu;
    GLuint A_log_size;

    float* B_log;
    GLuint B_log_gpu;
    GLuint B_log_size;

    GLuint dp_last_gpu;
    GLuint dp_last_size;

    GLuint output_gpu;
    GLuint output_size;

    GLuint computeProgram;

   public:
    float* dp_last;
    chcpy::hmm::hmm_predict_t* cpu;
    GPUContext* context;
    inline hmm_t(GPUContext* context, chcpy::hmm::hmm_predict_t* cpu) {
        this->cpu = cpu;
        this->context = context;

        CHECK();
        computeProgram = context->createComputeProgram(hmm_shader);  //创建gpu端程序
        CHECK();

        //创建GPU端变量
        A_log_size = cpu->N * cpu->N;
        A_log = new float[A_log_size];
        for (int i = 0; i < cpu->N; ++i) {
            for (int j = 0; j < cpu->N; ++j) {
                float val = cpu->A_log.at(i).at(j);
                A_log[i * cpu->N + j] = val;
            }
        }
        glGenBuffers(1, &A_log_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, A_log_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, A_log_size * sizeof(float), A_log, GL_STATIC_DRAW);

        B_log_size = cpu->N * cpu->M;
        B_log = new float[B_log_size];
        for (int i = 0; i < cpu->N; ++i) {
            for (int j = 0; j < cpu->M; ++j) {
                float val = cpu->B_log.at(i).at(j);
                B_log[i * cpu->M + j] = val;
            }
        }
        glGenBuffers(1, &B_log_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, B_log_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, B_log_size * sizeof(float), B_log, GL_STATIC_DRAW);

        dp_last_size = cpu->N;
        dp_last = new float[dp_last_size];
        for (int i = 0; i < dp_last_size; ++i) {
            dp_last[i] = 0;
        }
        glGenBuffers(1, &dp_last_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dp_last_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, dp_last_size * sizeof(float), dp_last, GL_STATIC_DRAW);

        output_size = cpu->N * 2;
        glGenBuffers(1, &output_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, output_size * sizeof(float), NULL, GL_STATIC_DRAW);
    }
    inline void run(GLint seq_i, const std::function<void(float*)>& callback) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A_log_gpu);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, B_log_gpu);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dp_last_gpu);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, dp_last_size * sizeof(float), dp_last);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dp_last_gpu);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, output_gpu);

        glUseProgram(computeProgram);
        CHECK();

        //创建uniform变量
        int gpu_M = glGetUniformLocation(computeProgram, "M");
        glUniform1i(gpu_M, cpu->M);
        int gpu_N = glGetUniformLocation(computeProgram, "N");
        glUniform1i(gpu_N, cpu->N);
        int gpu_seq_i = glGetUniformLocation(computeProgram, "seq_i");
        glUniform1i(gpu_seq_i, seq_i);

        glDispatchCompute(cpu->N, 1, 1);  //执行
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_gpu);
        float* pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, output_size * sizeof(float), GL_MAP_READ_BIT);
        CHECK();
        callback(pOut);

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    inline ~hmm_t() {
        glDeleteProgram(computeProgram);
        delete[] A_log;
        delete[] B_log;
        delete[] dp_last;
    }
};

template <typename T>
concept hmm_c = requires(T a) {
    a.dp_last;
    a.cpu;
    a.run(GLint(), [](float*) {});
    chcpy::hmm::hmm_predict_t(*a.cpu);
};
}  // namespace gpu

namespace hmm {

template <gpu::hmm_c h>
inline void predict(                 //维特比算法，获得最优切分路径
    h& gpu,                          //hmm对象
    const melody_t& seq,             //seq须先用melody2seq预处理
    std::vector<int>& best_sequence  //输出
) {
#ifdef CHCPY_DEBUG
    clock_t startTime, endTime;
    startTime = clock();  //计时开始
#endif
    auto& self = *gpu.cpu;
    auto T = seq.size();
    constexpr auto log0 = -std::numeric_limits<float>::infinity();
    std::vector<std::vector<float>> dp(T, std::vector<float>(self.N, 0));
    std::vector<std::vector<int>> ptr(T, std::vector<int>(self.N, 0));

    for (size_t j = 0; j < self.N; ++j) {
        float startval = self.P_log.at(j) + self.B_log.at(j).at(seq.at(0));
        dp.at(0).at(j) = startval;
        gpu.dp_last[j] = startval;
    }
    for (size_t i = 1; i < T; ++i) {
        auto& dpi = dp.at(i);
        /*
        for (size_t j = 0; j < self.N; ++j) {
            dpi.at(j) = log0;
            float base = self.B_log.at(j).at(seq.at(i));
            for (size_t k = 0; k < self.N; ++k) {
                float score = base + dp.at(i - 1).at(k) + self.A_log.at(k).at(j);
                if (score != log0 && score > dpi.at(j)) {
                    dpi.at(j) = score;
                    ptr.at(i).at(j) = k;
                }
            }
        }
        */
        gpu.run(seq.at(i), [&](float* output) {
            for (size_t j = 0; j < self.N; ++j) {
                float dpi_val = output[j * 2];
                int ptr_val = output[j * 2 + 1];
                //printf("%d\n",ptr_val);
                gpu.dp_last[j] = dpi_val;
                dpi.at(j) = dpi_val;
                ptr.at(i).at(j) = ptr_val;
            }
        });
    }
    best_sequence.resize(T);
    best_sequence.at(T - 1) = argmax(dp.at(T - 1).begin(), dp.at(T - 1).end());
    for (int i = T - 2; i >= 0; --i) {  //这里不能用size_t，否则将导致下溢出，造成死循环
        best_sequence.at(i) = ptr.at(i + 1).at(best_sequence.at(i + 1));
    }
#ifdef CHCPY_DEBUG
    endTime = clock();  //计时结束
    printf("\nhmm用时%f秒\n", (float)(endTime - startTime) / CLOCKS_PER_SEC);
#endif
}

}  // namespace hmm
}  // namespace chcpy