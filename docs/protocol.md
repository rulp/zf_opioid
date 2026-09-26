# Run protocol

How one self-administration run is carried out on our rig, and what to record so the pipeline can
analyse it. The assay is adapted from
[Bossé & Peterson (2017)](https://doi.org/10.1016/j.bbr.2017.08.001); see the README for how our
set-up differs.

## Fish

- Keep several groups of 20–30 fish, all trained to hydrocodone self-administration.
- Use one group per day, rotating through the groups. Rest each group for 7 days between uses.
- Individual fish are not tracked. The statistical unit is the **run**, not the fish.
- Re-verify addiction state periodically with no-compound runs, compared against the group's
  post-training baseline. These runs are not part of an analysis set.

## One run

1. **Draw 5 fish** at random from the day's group.
2. **Pre-incubate for 1 hour** in 150 mL of water with 50 µL of compound stock in DMSO.
   - A vehicle run gets 50 µL of DMSO alone (about 0.033%).
   - Keep the volume constant across conditions and record the molar concentration.
3. **Transfer the 5 fish to the rig for 50 minutes.** Start the rig software; it writes the run
   folder (`YYYYMMDD_HHMMSS_...`) with the two event logs, `params.json` and one JPEG per active
   trigger.
   - The compound is present only during pre-incubation, never in the rig water.
   - The rig water flows slowly, about one tank turnover per run.
4. **Move the fish to a recovery tank.** Do not reuse them the same day.

## Deaths and faults

- **If any fish dies, discard the whole run** and re-run the compound at a lower concentration. In
  `metadata.csv`, set `exclude` to `true` and give the reason.
- A run with a known equipment fault (for example, a bugged pump) can be kept: note it in the
  `flag` column. It is marked on the figures, and a refit without flagged runs is reported.
- Write down exclusions and deaths for each compound. They show where a compound's usable range
  ends.

## Planning runs for analysis

The pipeline compares runs within each day, against that day's vehicle runs. For results to be
usable:
- **Run vehicle (DMSO) on every experimental day**, ideally at least 2 vehicle runs per day. Day
  centring needs days with 2 or more.
- **Alternate the order** of vehicle and treated runs across days, so vehicle is not always first.
- **Spread each compound's replicates over several days.** A compound tested on a single day
  cannot be separated from that day's effect.
- **Aim for more than 3 runs per condition.** At n = 3 only near-total abolition of drug-seeking
  is detectable.
- Include positive controls (for example MK801, ketamine, naloxone or methadone) and mark them in
  `metadata.csv`.
- A run needs at least 20 active visits between 10 s and 2900 s to get pose features. Quieter runs
  are kept for the event features but not scored in the clustering.

## Not covered here

The following are set by the lab's own SOPs and are not specified by this pipeline:
- fish strain, age and sex;
- tank dimensions and water depth;
- water temperature and chemistry;
- the hydrocodone dose per trigger;
- the training protocol;
- the absolute flow rate;
- feeding times relative to runs.
