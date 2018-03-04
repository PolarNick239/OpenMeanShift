#pragma once

#ifdef _WIN32
#include <time.h>
#else

#include <sys/time.h>

#endif

class performance_timer {
protected:
#ifdef _WIN32
    typedef clock_t timer_type;
#else
    typedef struct timeval timer_type;
#endif

    double counter_;
    timer_type start_;
    int is_running_;

public:
    performance_timer(bool paused = false)
    {
        counter_ = 0;
        is_running_ = 0;
        if (!paused)
            start();
    }

    void start()
    {
        if (is_running_) return;

        start_ = measure();
        is_running_ = 1;
    }

    void stop()
    {
        if (!is_running_) return;

        counter_ += diff(start_, measure());
        is_running_ = 0;
    }

    void reset()
    {
        counter_ = 0;
        is_running_ = 0;
    }

    void restart()
    {
        reset();
        start();
    }

    double elapsed() const
    {
        double tm = counter_;

        if (is_running_)
            tm += diff(start_, measure());

        if (tm < 0)
            tm = 0;

        return tm;
    }

protected:
    static timer_type measure()
    {
        timer_type tm;
#ifdef _WIN32
        tm = clock();
#else
        ::gettimeofday(&tm, 0);
#endif
        return tm;
    }

    static double diff(const timer_type &start, const timer_type &end)
    {
#ifdef _WIN32
        return (double) (end - start) / (double) CLOCKS_PER_SEC;
#else
        long secs = end.tv_sec - start.tv_sec;
        long usecs = end.tv_usec - start.tv_usec;

        return (double) secs + (double) usecs / 1000000.0;
#endif
    }
};
