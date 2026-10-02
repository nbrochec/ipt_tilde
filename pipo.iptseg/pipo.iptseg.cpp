#include "PiPoIPTSeg.h"
#include "MaxPiPo.h"

#ifdef PIPO_MAX_WITH_DOC
#define NUM_PIPO_ATTRS 0
static const char *attrNames[NUM_PIPO_ATTRS] = {};
static const char *attrDescriptions[NUM_PIPO_ATTRS] = {};
PIPO_MAX_CLASS("iptseg", PiPoIPTSeg, "Segment and label IPT class distributions", "", NUM_PIPO_ATTRS, attrNames, attrDescriptions);
#else
PIPO_MAX_CLASS("iptseg", PiPoIPTSeg);
#endif
