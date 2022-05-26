#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include "ActivationFunction.h"
#include <string>

namespace std
{
    class ActivationFunctions
    {
    public:
        static ActivationFunction* findByName(string aName);
    };
}
#endif