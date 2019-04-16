#ifndef __KNEARESTNEIGHBORSCLASSIFIER_ENUMS_H
#define __KNEARESTNEIGHBORSCLASSIFIER_ENUMS_H
enum MLKNearestNeighborsClassifierClassLabels: int {
    MLKNearestNeighborsClassifierClassLabels_stringClassLabels = 100,
    MLKNearestNeighborsClassifierClassLabels_int64ClassLabels = 101,
    MLKNearestNeighborsClassifierClassLabels_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLKNearestNeighborsClassifierClassLabels_Name(MLKNearestNeighborsClassifierClassLabels x) {
    switch (x) {
        case MLKNearestNeighborsClassifierClassLabels_stringClassLabels:
            return "MLKNearestNeighborsClassifierClassLabels_stringClassLabels";
        case MLKNearestNeighborsClassifierClassLabels_int64ClassLabels:
            return "MLKNearestNeighborsClassifierClassLabels_int64ClassLabels";
        case MLKNearestNeighborsClassifierClassLabels_NOT_SET:
            return "INVALID";
    }
}

#endif
