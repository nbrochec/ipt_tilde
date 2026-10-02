// -*- mode: c++; c-basic-offset:2 -*-

// pipo.iptseg: turns the class distributions produced by pipo.ipt into labelled
// segments (markers with a duration), for automatic annotation in MuBu.
//
//   mubu.process <c> audio ipt:iptseg @name markers ...
//   mubu.process <c> ipt iptseg @name markers ...      (re-segment a stored ipt track)
//   pipo~ ipt:iptseg ...                                 (one frame per segment, at its end)
//
// The filtering follows the IPT Vamp plug-in (Ircam-Partiels/ipt-vamp-plugin):
// each frame votes for its argmax class and consecutive frames of the same class
// are merged into one segment, reported with its score and its number of
// occurrences. On top of that:
//   - @confidence  frames whose top probability is below it do not vote (the
//                  equivalent of -1 / no_confidence in ipt~; @ipt.confidence
//                  has no effect since pipo.ipt only outputs the distribution)
//   - @mindur      a class change is only accepted once the new class has lasted
//                  that long; shorter runs are absorbed into the current segment
//   - @gap         a hole with no voting frame (energy gate, low confidence)
//                  longer than this ends the segment (the Vamp plug-in merges
//                  across silences; set a large gap to get that behaviour)
//
// Output, one time-tagged frame per segment at its start time:
//   Class (index into the input columns), Score (mean top probability),
//   Occurrences (voting frames), Duration (ms)
//
// No libipt, no torch: this module only reads its input stream, so it can run on
// the raw output of pipo.ipt or on a distribution track stored in a mubu.

#include "PiPo.h"
#include <algorithm>
#include <cmath>
#include <vector>

class PiPoIPTSeg : public PiPo
{
private:
  struct Run
  {
    int    cls   = -1;   // -1: no run
    double start = 0;    // time of the first voting frame (ms)
    double last  = 0;    // time of the last voting frame (ms)
    double score = 0;    // sum of the top probabilities
    int    count = 0;    // voting frames
  };

  Run    seg_;            // current (confirmed) segment
  Run    cand_;           // challenger class, not yet confirmed
  double frame_period_ = 0;   // ms between input frames, from the stream rate

  static constexpr int NUM_OUT = 4;

  void clear_state ()
  {
    seg_  = Run();
    cand_ = Run();
  }

  double gap_ms ()
  {
    // never shorter than the input frame spacing, or every frame would be a hole
    return std::max<double>(gap_attr_.get(), 1.5 * frame_period_);
  }

  // A run covers its voting frames plus one frame period after the last one.
  double run_duration (const Run &r) const
  {
    return r.last - r.start + frame_period_;
  }

  int emit (const Run &r, double end)
  {
    if (r.cls < 0  ||  r.count == 0)
      return PIPO_OK;

    PiPoValue out[NUM_OUT];
    out[0] = static_cast<PiPoValue>(r.cls);
    out[1] = static_cast<PiPoValue>(r.score / r.count);
    out[2] = static_cast<PiPoValue>(r.count);
    out[3] = static_cast<PiPoValue>(end - r.start);
    return propagateFrames(r.start + offset_attr_.get(), 1.0, out, NUM_OUT, 1);
  }

  // Close everything at the end of a hole or of the stream.
  int close_all ()
  {
    int ret = PIPO_OK;
    if (seg_.cls >= 0)
    {
      // an unconfirmed challenger after the segment was too short: absorbed
      double end = std::max(seg_.last, cand_.cls >= 0 ? cand_.last : seg_.last) + frame_period_;
      ret = emit(seg_, end);
    }
    else if (cand_.cls >= 0  &&  run_duration(cand_) >= mindur_attr_.get())
      ret = emit(cand_, cand_.last + frame_period_);

    clear_state();
    return ret;
  }

  int vote (double time, int cls, double score)
  {
    int ret = PIPO_OK;

    // a hole since the last voting frame ends the current segment
    double last = std::max(seg_.cls >= 0 ? seg_.last : -1e18, cand_.cls >= 0 ? cand_.last : -1e18);
    if (time - last > gap_ms ())
      ret = close_all();

    if (cls == seg_.cls)
    {
      // back to the current class: the challenger was a blip, absorb it
      seg_.last   = time;
      seg_.score += score;
      seg_.count += 1;
      cand_ = Run();
      return ret;
    }

    if (cls != cand_.cls)
    {
      cand_       = Run();
      cand_.cls   = cls;
      cand_.start = time;
    }
    cand_.last   = time;
    cand_.score += score;
    cand_.count += 1;

    if (run_duration(cand_) >= mindur_attr_.get())
    {
      // the challenger has lasted long enough: it takes over at its first frame
      int r = emit(seg_, cand_.start);
      if (r != PIPO_OK)
        ret = r;
      seg_  = cand_;
      cand_ = Run();
    }
    return ret;
  }

public:
  PiPoScalarAttr<float> confidence_attr_;
  PiPoScalarAttr<float> mindur_attr_;
  PiPoScalarAttr<float> gap_attr_;
  PiPoScalarAttr<float> offset_attr_;

  PiPoIPTSeg (Parent *parent, PiPo *receiver = NULL)
  : PiPo(parent, receiver),
    confidence_attr_ (this, "confidence", "Frames whose top probability is below this value do not vote", false, 0.0),
    mindur_attr_     (this, "mindur", "Minimum duration (ms) a new class must last before it starts a segment", false, 0.0),
    gap_attr_        (this, "gap", "A hole (ms) without any voting frame longer than this ends the current segment", false, 50.0),
    offset_attr_     (this, "offset", "Time offset (ms) added to the segment times", false, 0.0)
  { }

  int streamAttributes (bool hasTimeTags, double rate, double offset,
                        unsigned int width, unsigned int height,
                        const char **labels, bool hasVarSize,
                        double domain, unsigned int maxFrames)
  {
    frame_period_ = rate > 0 ? 1000.0 / rate : 0;
    clear_state();

    static const char *out_labels[NUM_OUT] = { "Class", "Score", "Occurrences", "Duration" };
    return propagateStreamAttributes(true, rate, offset, NUM_OUT, 1, out_labels, false, 0, 1);
  }

  int reset (void)
  {
    clear_state();
    return propagateReset();
  }

  int frames (double time, double weight, PiPoValue *values, unsigned int size, unsigned int num)
  {
    int ret = PIPO_OK;
    float confidence = confidence_attr_.get();

    for (unsigned int i = 0; i < num; i++, values += size)
    {
      if (size == 0)
        continue;

      PiPoValue *top = std::max_element(values, values + size);
      if (*top < confidence)
        continue;   // does not vote; a long enough run of these ends the segment

      // frames in a block share the time of the first one, spaced by the period
      int r = vote(time + i * frame_period_, static_cast<int>(top - values), *top);
      if (r != PIPO_OK)
        ret = r;
    }
    return ret;
  }

  int finalize (double inputEnd)
  {
    int ret = close_all();
    int r = propagateFinalize(inputEnd);
    return ret != PIPO_OK ? ret : r;
  }
};
