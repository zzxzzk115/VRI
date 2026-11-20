// android_main.cpp - GameActivity entry for the VRI examples on Android.
//
// GameActivity runs android_main on its own thread. It routes the example's stdout/stderr (VRI's
// diagnostics, the "rendering"/"frames presented" lines) to logcat under the tag "VRI", turns the
// launch Intent's string extras into the environment variables the desktop examples read, then
// runs the example's main (renamed by example_app.h) and finishes the activity when it returns:
//
//     adb shell am start -n org.vri.example.cube/com.google.androidgamesdk.GameActivity \
//         --es VRI_API opengl --es VRI_MAX_FRAMES 300
#include "example_app.h"

#include <android/log.h>
#include <game-activity/GameActivity.h>
#include <jni.h>
#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

namespace
{
    int g_logPipe[2] = {-1, -1};

    // Forwards every line written to the pipe to logcat.
    void* ForwardToLogcat(void*)
    {
        char   line[1024];
        size_t used = 0;
        char   byte = 0;
        while (read(g_logPipe[0], &byte, 1) == 1)
        {
            if (byte == '\n' || used == sizeof(line) - 1)
            {
                line[used] = '\0';
                __android_log_write(ANDROID_LOG_INFO, "VRI", line);
                used = 0;
                if (byte == '\n')
                    continue;
            }
            line[used++] = byte;
        }
        return nullptr;
    }

    // Android discards a native process's stdout and stderr.
    void RedirectStdioToLogcat()
    {
        if (pipe(g_logPipe) != 0)
            return;
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        std::setvbuf(stderr, nullptr, _IONBF, 0);
        dup2(g_logPipe[1], STDOUT_FILENO);
        dup2(g_logPipe[1], STDERR_FILENO);
        pthread_t thread;
        pthread_create(&thread, nullptr, ForwardToLogcat, nullptr);
        pthread_detach(thread);
    }

    // The environment variables the examples read (examples/common/example_app.h), set from the
    // launch Intent's string extras of the same name.
    void ImportIntentEnvironment(GameActivity& activity)
    {
        static const char* const kNames[] = {"VRI_API", "VRI_MAX_FRAMES", "VRI_PRESENT"};

        // GameActivity has usually attached this thread already; detach only an attachment made here.
        JNIEnv*    env      = nullptr;
        const bool attached = activity.vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_EDETACHED;
        if (attached && activity.vm->AttachCurrentThread(&env, nullptr) != JNI_OK)
            return;
        jobject intent = env->CallObjectMethod(activity.javaGameActivity,
                                               env->GetMethodID(env->GetObjectClass(activity.javaGameActivity),
                                                                "getIntent",
                                                                "()Landroid/content/Intent;"));
        if (intent)
        {
            jmethodID getStringExtra = env->GetMethodID(
                env->GetObjectClass(intent), "getStringExtra", "(Ljava/lang/String;)Ljava/lang/String;");
            for (const char* name : kNames)
            {
                jstring key   = env->NewStringUTF(name);
                auto    value = static_cast<jstring>(env->CallObjectMethod(intent, getStringExtra, key));
                if (value)
                {
                    const char* text = env->GetStringUTFChars(value, nullptr);
                    setenv(name, text, 1);
                    env->ReleaseStringUTFChars(value, text);
                    env->DeleteLocalRef(value);
                }
                env->DeleteLocalRef(key);
            }
            env->DeleteLocalRef(intent);
        }
        if (env->ExceptionCheck())
            env->ExceptionClear();
        if (attached)
            activity.vm->DetachCurrentThread();
    }
} // namespace

extern "C" void android_main(android_app* app)
{
    RedirectStdioToLogcat();
    ImportIntentEnvironment(*app->activity);
    vriex::g_androidApp    = app;
    vriex::g_androidAssets = app->activity->assetManager;

    char  program[] = "vri-example";
    char* argv[]    = {program, nullptr};
    vriex::ExampleApp::ExitAndroid(vriExampleMain(1, argv));
}
