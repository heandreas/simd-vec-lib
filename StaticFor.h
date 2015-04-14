#ifndef STATICFOR_H
#define STATICFOR_H

#ifdef _MSVC_VER
#define Inline __forceinline
#else
#define Inline
#endif

#define begin_disable_warnings
#define end_disable_warnings

// source: http://www.codeproject.com/Articles/75423/Loop-Unrolling-over-Template-Arguments
template<int Begin, int End, int Step = 1>
//lambda unroller
struct StaticFor {
    template<typename Lambda>
    Inline static void step(const Lambda& func) {
        func(Begin);
        StaticFor<Begin+Step, End, Step>::step(func);
    }
};
//end of lambda unroller
template<int End, int Step>
struct StaticFor<End, End, Step> {
    template<typename Lambda>
    Inline static void step(const Lambda&) {
    }
};

#endif // STATICFOR_H
