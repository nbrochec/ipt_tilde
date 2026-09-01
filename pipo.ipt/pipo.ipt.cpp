
#include "PiPoIPT.h"
#include "MaxPiPo.h"
#include "ext_path.h"
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <climits>

// Resolve a (possibly relative) model filename against Max's search path.
// Declared in PiPoIPT.h; implemented here where the Max SDK is available.
std::string ipt_resolve_model_path (const char *name)
{
  if (name == NULL || name[0] == '\0')
    return "";

  // already readable as given (absolute, or relative to the cwd): use as-is
  if (access(name, R_OK) == 0)
    return name;

  // otherwise search Max's file search path
  char filename[MAX_PATH_CHARS];
  strncpy(filename, name, MAX_PATH_CHARS - 1);
  filename[MAX_PATH_CHARS - 1] = '\0';

  short    outvol  = 0;
  t_fourcc outtype = 0;
  if (locatefile_extended(filename, &outvol, &outtype, NULL, 0) == 0)
  {
    char abspath[MAX_PATH_CHARS];
    if (path_toabsolutesystempath(outvol, filename, abspath) == 0)
    {
      // canonicalize the /Volumes/<bootvol>/... form Max returns: torch
      // cannot open the symlinked path, but realpath() resolves it to /Users/...
      char real[PATH_MAX];
      if (realpath(abspath, real) != NULL)
        return real;
      return abspath;
    }
  }

  return name;  // not found: caller reports a clear "cannot read model file" error
}

#ifdef PIPO_MAX_WITH_DOC
#define NUM_PIPO_ATTRS 0
static const char *attrNames[NUM_PIPO_ATTRS] = {};
static const char *attrDescriptions[NUM_PIPO_ATTRS] = {};
PIPO_MAX_CLASS("ipt", PiPoIPT, "IPT playing style recognition", "", NUM_PIPO_ATTRS, attrNames, attrDescriptions);
#else
PIPO_MAX_CLASS("ipt", PiPoIPT);
#endif
