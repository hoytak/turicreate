#ifndef __NEARESTNEIGHBORS_ENUMS_H
#define __NEARESTNEIGHBORS_ENUMS_H
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

enum MLKNearestNeighborsClassifierWeightingScheme: int {
    MLKNearestNeighborsClassifierWeightingScheme_uniformWeighting = 200,
    MLKNearestNeighborsClassifierWeightingScheme_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLKNearestNeighborsClassifierWeightingScheme_Name(MLKNearestNeighborsClassifierWeightingScheme x) {
    switch (x) {
        case MLKNearestNeighborsClassifierWeightingScheme_uniformWeighting:
            return "MLKNearestNeighborsClassifierWeightingScheme_uniformWeighting";
        case MLKNearestNeighborsClassifierWeightingScheme_NOT_SET:
            return "INVALID";
    }
}

enum MLNearestNeighborsIndexIndexType: int {
    MLNearestNeighborsIndexIndexType_linearIndex = 100,
    MLNearestNeighborsIndexIndexType_singleKdTreeIndex = 110,
    MLNearestNeighborsIndexIndexType_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLNearestNeighborsIndexIndexType_Name(MLNearestNeighborsIndexIndexType x) {
    switch (x) {
        case MLNearestNeighborsIndexIndexType_linearIndex:
            return "MLNearestNeighborsIndexIndexType_linearIndex";
        case MLNearestNeighborsIndexIndexType_singleKdTreeIndex:
            return "MLNearestNeighborsIndexIndexType_singleKdTreeIndex";
        case MLNearestNeighborsIndexIndexType_NOT_SET:
            return "INVALID";
    }
}

enum MLNearestNeighborsIndexDistanceFunction: int {
    MLNearestNeighborsIndexDistanceFunction_squaredEuclideanDistance = 200,
    MLNearestNeighborsIndexDistanceFunction_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLNearestNeighborsIndexDistanceFunction_Name(MLNearestNeighborsIndexDistanceFunction x) {
    switch (x) {
        case MLNearestNeighborsIndexDistanceFunction_squaredEuclideanDistance:
            return "MLNearestNeighborsIndexDistanceFunction_squaredEuclideanDistance";
        case MLNearestNeighborsIndexDistanceFunction_NOT_SET:
            return "INVALID";
    }
}

#endif
