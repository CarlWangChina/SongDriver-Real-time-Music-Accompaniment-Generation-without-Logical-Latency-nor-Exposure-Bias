#pragma once
#include "gpu.h"
#include "hmmv3.h"
namespace chcpy {
namespace gpu {

constexpr char hmmv3_shader[] = R"(

#version 320 es

uniform int M;
uniform int N;
uniform int A2_len;
uniform int B2_len;
uniform int seq_i;  // seq.at(i)

layout(local_size_x = 1) in;

layout(binding = 0) readonly buffer Input1 {
    int data[];
    //格式：
    //[n*3+0] yz混合编码
    //[n*3+1] A2_log_ptr_x对应坐标
    //[n*3+2] 长度
}A2_log_ptr_yz;

layout(binding = 1) readonly buffer Input2 {
    int data[];
}A2_log_ptr_x;

layout(binding = 2) readonly buffer Input3 {
    float data[];
}A2_log;

layout(binding = 3) readonly buffer Input5 {
    int data[];
    //格式：
    //[n*3+0] xy混合编码
    //[n*3+1] B2_log_ptr_z对应坐标
    //[n*3+2] 长度
}B2_log_ptr_xy;

layout(binding = 4) readonly buffer Input6 {
    int data[];
}B2_log_ptr_z;

layout(binding = 5) readonly buffer Input7 {
    float data[];
}B2_log;

layout(binding = 6) readonly buffer Input8 {
    float data[];
}dp_last;

layout(binding = 7) writeonly buffer Output0 {
    float data[];
}output0;

int mergeInt16(int x, int y) {
    //将两个16为整数混合为一个int
    return (((x & 0x0000ffff) << 4) | (y & 0x0000ffff));
}

void main() {
    if(A2_len==0 || B2_len==0){
        return;
    }
    const float infinity = 1. / 0.;  //定义无限大常量
    int j = int(gl_GlobalInvocationID.x);
    int k1 = int(gl_GlobalInvocationID.y);

    int j_k1 = mergeInt16(j, k1);

    //定位A2_log的x坐标
    //二分查找A2_log_ptr_yz
    int s;
    int ptr_A2_yz = 0;
    int ptr_A2_yz_move = A2_len;
    for (int i = 0; i < 32; ++i) {
        int delta_yz = j_k1 - A2_log_ptr_yz.data[ptr_A2_yz * 3];
        s = sign(delta_yz);
        ptr_A2_yz += s * ptr_A2_yz_move;
        ptr_A2_yz_move = max(ptr_A2_yz_move / 2, 1);
        ptr_A2_yz = clamp(ptr_A2_yz, 0, max(A2_len - 1, 0));
    }
    bool A2_haveValue;
    int A2_begin = A2_log_ptr_yz.data[ptr_A2_yz * 3 + 1];
    int A2_num = A2_log_ptr_yz.data[ptr_A2_yz * 3 + 2];

    //if (A2_log_ptr_yz.data[ptr_A2_yz * 3] == j_k1) {
    //    A2_haveValue = true;
    //} else {
    //    A2_haveValue = false;
    //}
    A2_haveValue = (A2_log_ptr_yz.data[ptr_A2_yz * 3] == j_k1);

    //定位B2_log
    int ptr_B2_xy = 0;
    int ptr_B2_xy_move = B2_len;
    for (int i = 0; i < 32; ++i) {
        int delta_xy = j_k1 - B2_log_ptr_xy.data[ptr_B2_xy * 3];
        s = sign(delta_xy);
        ptr_B2_xy += s * ptr_B2_xy_move;
        ptr_B2_xy_move = max(ptr_B2_xy_move / 2, 1);
        ptr_B2_xy = clamp(ptr_B2_xy, 0, max(B2_len - 1, 0));
    }
    bool B2_haveValue;
    int B2_begin = B2_log_ptr_xy.data[ptr_B2_xy * 3 + 1];
    int B2_num = B2_log_ptr_xy.data[ptr_B2_xy * 3 + 2];
    //if (B2_log_ptr_xy.data[ptr_B2_xy * 3] == j_k1) {
    //    B2_haveValue = true;
    //} else {
    //    B2_haveValue = false;
    //}
    B2_haveValue = (B2_log_ptr_xy.data[ptr_B2_xy * 3] == j_k1);
    // B2进一步查找
    int ptr_B2_z = 0;
    int ptr_B2_z_move = B2_num;
    for (int i = 0; i < 16; ++i) {
        int delta_z = seq_i - B2_log_ptr_z.data[ptr_B2_z + B2_begin];
        s = sign(delta_z);
        ptr_B2_z += s * ptr_B2_z_move;
        ptr_B2_z_move = max(ptr_B2_z_move / 2, 1);
        ptr_B2_z = clamp(ptr_B2_z, 0, max(B2_num - 1, 0));
    }
    float B2_log_val;
    if (B2_log_ptr_xy.data[ptr_B2_z + B2_begin] == seq_i && B2_haveValue) {
        B2_log_val = B2_log.data[ptr_B2_z + B2_begin];
    } else {
        B2_log_val = -infinity;
    }

    //开始正式搜索
    float max_val = -infinity;
    int max_path = -1;
    int A2_log_index = 0;
    for (int k2 = 0; k2 < N; ++k2) {
        float A2_log_value = -infinity;
        //搜索A2
        if (A2_haveValue) {
            if (A2_log_ptr_x.data[A2_begin + A2_log_index] < k2) {
                A2_log_index += clamp(A2_log_index + 1, 0, max(A2_num - 1, 0));
            } else if (A2_log_ptr_x.data[A2_begin + A2_log_index] == k2) {
                A2_log_value = A2_log.data[A2_begin + A2_log_index];
                A2_log_index += clamp(A2_log_index + 1, 0, max(A2_num - 1, 0));
            }
        }
        float val = dp_last.data[j + k2 * N] + A2_log_value + B2_log_val;
        if (val > max_val) {
            max_val = val;
            max_path = k2;
        }
    }
    output0.data[(j * N + k1) * 2] = max_val;
    output0.data[(j * N + k1) * 2 + 1] = float(max_path);
}

)";

inline int mergeInt16(int x, int y) {
    //将两个16为整数混合为一个int
    return (((x & 0x0000ffff) << 4) | (y & 0x0000ffff));
}

class hmmv3_t {
   private:
    //稀疏矩阵A2
    std::vector<GLint> A2_log_ptr_yz;
    GLuint A2_log_ptr_yz_gpu;
    std::vector<GLint> A2_log_ptr_x;
    GLuint A2_log_ptr_x_gpu;
    std::vector<float> A2_log;
    GLuint A2_log_gpu;
    int A2_len;

    //稀疏矩阵B2
    std::vector<GLint> B2_log_ptr_xy;
    GLuint B2_log_ptr_xy_gpu;
    std::vector<GLint> B2_log_ptr_z;
    GLuint B2_log_ptr_z_gpu;
    std::vector<float> B2_log;
    GLuint B2_log_gpu;
    int B2_len;

    GLuint dp_last_gpu;
    GLuint dp_last_size;

    GLuint output_gpu;
    GLuint output_size;

    GLuint computeProgram;

    inline void genA2GPUBuffer() {
        std::map<int, std::map<int, float>> m;
        for (auto& it : cpu->A2_log.data) {
            int x = std::get<0>(it.first);
            int y = std::get<1>(it.first);
            int z = std::get<2>(it.first);
            int yz = mergeInt16(y, z);
            m[yz][x] = it.second;
        }
        int index = 0;
        A2_len = m.size();
        for (auto& it_yz : m) {
            A2_log_ptr_yz.push_back(it_yz.first);
            A2_log_ptr_yz.push_back(index);
            A2_log_ptr_yz.push_back(it_yz.second.size());
            index += it_yz.second.size();
            for (auto& it_x : it_yz.second) {
                A2_log_ptr_x.push_back(it_x.first);
                A2_log.push_back(it_x.second);
            }
        }

        //上传gpu
        glGenBuffers(1, &A2_log_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, A2_log_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     A2_log.size() * sizeof(float),
                     A2_log.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &A2_log_ptr_yz_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, A2_log_ptr_yz_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     A2_log_ptr_yz.size() * sizeof(float),
                     A2_log_ptr_yz.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &A2_log_ptr_x_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, A2_log_ptr_x_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     A2_log_ptr_x.size() * sizeof(float),
                     A2_log_ptr_x.data(),
                     GL_STATIC_DRAW);
    }
    inline void genB2GPUBuffer() {
        std::map<int, std::map<int, float>> m;
        for (auto it : cpu->B2_log.data) {
            int x = std::get<0>(it.first);
            int y = std::get<1>(it.first);
            int z = std::get<2>(it.first);
            int xy = mergeInt16(x, y);
            m[xy][z] = it.second;
        }
        int index = 0;
        B2_len = m.size();
        for (auto& it_xy : m) {
            B2_log_ptr_xy.push_back(it_xy.first);
            B2_log_ptr_xy.push_back(index);
            B2_log_ptr_xy.push_back(it_xy.second.size());
            index += it_xy.second.size();
            for (auto& it_z : it_xy.second) {
                B2_log_ptr_z.push_back(it_z.first);
                B2_log.push_back(it_z.second);
            }
        }

        //上传gpu
        glGenBuffers(1, &B2_log_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, B2_log_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     B2_log.size() * sizeof(float),
                     B2_log.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &B2_log_ptr_xy_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, B2_log_ptr_xy_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     B2_log_ptr_xy.size() * sizeof(float),
                     B2_log_ptr_xy.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &B2_log_ptr_z_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, B2_log_ptr_z_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                     B2_log_ptr_z.size() * sizeof(float),
                     B2_log_ptr_z.data(),
                     GL_STATIC_DRAW);
    }

   public:
    float* dp_last;
    chcpy::hmm::hmmv3_predict_t* cpu;
    GPUContext* context;
    inline hmmv3_t(GPUContext* context, chcpy::hmm::hmmv3_predict_t* cpu) {
        this->cpu = cpu;
        this->context = context;

        CHECK();
        computeProgram = context->createComputeProgram(hmmv3_shader);  //创建gpu端程序
        CHECK();

        //创建GPU端变量

        genA2GPUBuffer();
        genB2GPUBuffer();
        printf("A2_len=%d B2_len=%d\n", A2_len, B2_len);

        dp_last_size = cpu->N * cpu->N;
        dp_last = new float[dp_last_size];
        for (int i = 0; i < dp_last_size; ++i) {
            dp_last[i] = 0;
        }
        glGenBuffers(1, &dp_last_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dp_last_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, dp_last_size * sizeof(float), dp_last, GL_STATIC_DRAW);

        output_size = cpu->N * cpu->N * 2;
        glGenBuffers(1, &output_gpu);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_gpu);
        glBufferData(GL_SHADER_STORAGE_BUFFER, output_size * sizeof(float), NULL, GL_STATIC_DRAW);
    }
    inline void run(GLint seq_i, const std::function<void(float*)>& callback) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A2_log_ptr_yz_gpu);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A2_log_ptr_x_gpu);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, A2_log_gpu);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, B2_log_ptr_xy_gpu);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, B2_log_ptr_z_gpu);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, B2_log_gpu);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, dp_last_gpu);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, dp_last_size * sizeof(float), dp_last);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, dp_last_gpu);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, output_gpu);

        glUseProgram(computeProgram);
        CHECK();

        //创建uniform变量
        int gpu_M = glGetUniformLocation(computeProgram, "M");
        glUniform1i(gpu_M, cpu->M);
        int gpu_N = glGetUniformLocation(computeProgram, "N");
        glUniform1i(gpu_N, cpu->N);
        int gpu_seq_i = glGetUniformLocation(computeProgram, "seq_i");
        glUniform1i(gpu_seq_i, seq_i);
        int gpu_A2_len = glGetUniformLocation(computeProgram, "A2_len");
        glUniform1i(gpu_A2_len, A2_len);
        int gpu_B2_len = glGetUniformLocation(computeProgram, "B2_len");
        glUniform1i(gpu_B2_len, B2_len);

        glDispatchCompute(cpu->N, cpu->N, 1);  //执行
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_gpu);
        float* pOut = (float*)glMapBufferRange(
            GL_SHADER_STORAGE_BUFFER, 0,
            output_size * sizeof(float),
            GL_MAP_READ_BIT);
        CHECK();
        callback(pOut);

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    inline ~hmmv3_t() {
        glDeleteProgram(computeProgram);
        delete[] dp_last;
    }
};

template <typename T>
concept hmmv3_c = requires(T a) {
    a.dp_last;
    a.cpu;
    a.run(GLint(), [](float*) {});
    chcpy::hmm::hmmv3_predict_t(*a.cpu);
};

}  // namespace gpu

namespace hmm {

template <gpu::hmmv3_c h>
inline void predict(                 //维特比算法，获得最优切分路径
    h& gpu,                          //hmm对象
    const melody_t& seq,             //seq须先用melody2seq预处理
    std::vector<int>& best_sequence  //输出
) {
    auto& self = *gpu.cpu;
    auto T = seq.size();
    constexpr auto log0 = -std::numeric_limits<float>::infinity();

    std::vector<SMat2D_predict> dp(T, SMat2D_predict(log0));
    std::vector<SMat2D_predict> ptr(T, SMat2D_predict(-1));

#pragma omp parallel for
    for (int x = 0; x < self.N; x++) {
        for (int y = 0; y < self.N; y++) {
            float val = self.P_log.at(x) + self.B1_log(x, seq.at(0)) +
                        self.A1_log(x, y) + self.B2_log(x, y, seq.at(1));
            //dp[1][x][y] = val;
            //ptr[1][x][y] = -1;
            gpu.dp_last[x * self.N + y] = val;
        }
    }

    for (int i = 2; i < T; i++) {
        int num_3 = seq.at(i);
        auto& dpi = dp.at(i);
        auto& dp_last = dp.at(i - 1);
        /*
        for (int j = 0; j < self.N; j++) {
            for (int k1 = 0; k1 < self.N; k1++) {
                float max_val = log0;
                int max_path = -1;
                for (int k2 = 0; k2 < self.N; k2++) {
                    float val = dp_last[k2][j] + self.A2_log(k2, j, k1);
                    if (val > max_val) {
                        max_val = val;
                        max_path = k2;
                    }
                }
                dpi[j][k1] = max_val + self.B2_log(j, k1, num_3);
                ptr[i][j][k1] = max_path;
            }
        }
        */
        gpu.run(num_3, [&](float* output) {
            for (int j = 0; j < self.N; j++) {
                for (int k1 = 0; k1 < self.N; k1++) {
                    float dpival = output[(j * self.N + k1) * 2];
                    int ptrval = output[(j * self.N + k1) * 2 + 1];
                    if (dpival != log0) {
                        dpi.set(j, k1, dpival);
                    }
                    if (ptrval != -1) {
                        ptr[i].set(j, k1, ptrval);
                    }
                    //if (ptrval != 0) {
                    //    printf("%d %d %d %d\n", i, j, k1, ptrval);
                    //}
                    gpu.dp_last[j * self.N + k1] = dpival;
                }
            }
        });
    }

    best_sequence.resize(T);
    //argmax 2d
    float max_val = log0;
    int max_path_i = -1;
    int max_path_j = -1;
    auto& dpi = dp.at(T - 1);
    for (int i = 0; i < self.N; i++) {
        for (int j = 0; j < self.N; j++) {
            if (dpi(i, j) > max_val) {
                max_val = dpi(i, j);
                max_path_i = i;
                max_path_j = j;
            }
        }
    }
    best_sequence.at(T - 1) = max_path_j;
    best_sequence.at(T - 2) = max_path_i;

    for (int t = T - 1; t > 1; t--) {
        int max_path_k = ptr[t](max_path_i, max_path_j);
        best_sequence.at(t - 2) = max_path_k;
        max_path_j = max_path_i;
        max_path_i = max_path_k;
    }
}

}  // namespace hmm
}  // namespace chcpy