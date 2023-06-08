#pragma once
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#include <stdio.h>
#include <stdlib.h>
#include "hmm.h"
namespace chcpy::gpu {

#define CHECK()                                                            \
    {                                                                      \
        GLenum err = glGetError();                                         \
        if (err != GL_NO_ERROR) {                                          \
            printf(__FILE__ ":%d glGetError returns %d\n", __LINE__, err); \
        }                                                                  \
    }
class GPUContext {
   private:
    EGLContext context;
    EGLDisplay dpy;

   public:
    inline GPUContext() {
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy == EGL_NO_DISPLAY) {
            printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
            return;
        }

        EGLint majorVersion;
        EGLint minorVersion;
        EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
        if (returnValue != EGL_TRUE) {
            printf("eglInitialize failed\n");
            return;
        }

        EGLConfig cfg;
        EGLint count;
        EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_NONE};
        if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
            printf("eglChooseConfig failed\n");
            return;
        }

        EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
        if (context == EGL_NO_CONTEXT) {
            printf("eglCreateContext failed\n");
            return;
        }
        returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
        if (returnValue != EGL_TRUE) {
            printf("eglMakeCurrent failed returned %d\n", returnValue);
            return;
        }
    }
    inline GLuint loadShader(GLenum shaderType, const char* pSource) {
        GLuint shader = glCreateShader(shaderType);
        if (shader) {
            glShaderSource(shader, 1, &pSource, NULL);
            glCompileShader(shader);
            GLint compiled = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
            if (!compiled) {
                GLint infoLen = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
                if (infoLen) {
                    char* buf = (char*)malloc(infoLen);
                    if (buf) {
                        glGetShaderInfoLog(shader, infoLen, NULL, buf);
                        fprintf(stderr, "Could not compile shader %d:\n%s\n",
                                shaderType, buf);
                        free(buf);
                    }
                    glDeleteShader(shader);
                    shader = 0;
                }
            }
        }
        return shader;
    }
    inline GLuint createComputeProgram(const char* pComputeSource) {
        GLuint computeShader = loadShader(GL_COMPUTE_SHADER, pComputeSource);
        if (!computeShader) {
            return 0;
        }

        GLuint program = glCreateProgram();
        if (program) {
            glAttachShader(program, computeShader);
            glLinkProgram(program);
            GLint linkStatus = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
            if (linkStatus != GL_TRUE) {
                GLint bufLength = 0;
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
                if (bufLength) {
                    char* buf = (char*)malloc(bufLength);
                    if (buf) {
                        glGetProgramInfoLog(program, bufLength, NULL, buf);
                        fprintf(stderr, "Could not link program:\n%s\n", buf);
                        free(buf);
                    }
                }
                glDeleteProgram(program);
                program = 0;
            }
        }
        return program;
    }
    inline ~GPUContext() {
        eglDestroyContext(dpy, context);
        eglTerminate(dpy);
    }
};

}  // namespace chcpy::gpu