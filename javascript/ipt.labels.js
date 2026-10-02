// ipt.labels.js: write class names as labels on a segment track made by pipo.iptseg.
//
// PiPo frames only carry numbers, so pipo.iptseg stores the class index in its
// first column. This script turns those indices into MuBu labels:
//
//   1. send it the class names, e.g. from the distribution track written by pipo.ipt
//        [mubu.track <c> <buf> ipt]  <- getmatrixcolnames
//      or type them: classnames ord pizz flatt ...
//   2. send it the segment matrix
//        [mubu.track <c> <buf> markers]  <- getmatrix
//   3. connect the outlet back to that same mubu.track: it receives
//        extradata label
//        setlabel 0 <name of segment 0> <name of segment 1> ...
//
// Indices without a name (or before any class names were received) are written
// as their number.

autowatch = 1;
inlets = 1;
outlets = 1;

var names = [];
var numcols = 4;   // columns of a pipo.iptseg frame: Class Score Occurrences Duration
var classcol = 0;

function cols (n) { numcols = Math.max(1, n); }

function classnames ()
{
  names = arrayfromargs(arguments);
}

function label_of (index)
{
  var i = Math.round(index);
  return (i >= 0 && i < names.length) ? names[i] : String(i);
}

function write_labels (values)
{
  if (values.length < numcols)
    return;

  var labels = [];
  for (var i = classcol; i < values.length; i += numcols)
    labels.push(label_of(values[i]));

  outlet(0, "extradata", "label");
  outlet(0, ["setlabel", 0].concat(labels));
}

function list ()
{
  write_labels(arrayfromargs(arguments));
}

// mubu.track replies are prefixed by a selector: symbols are class names
// (getmatrixcolnames), numbers are matrix values (getmatrix).
function anything ()
{
  var args = arrayfromargs(arguments);
  var numeric = args.length > 0;
  for (var i = 0; i < args.length; i++)
    if (typeof args[i] !== "number") { numeric = false; break; }

  if (numeric)
    write_labels(args);
  else if (args.length > 0)
    names = args;
}
