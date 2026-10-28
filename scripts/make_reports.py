"""Write the plain-text reports: parameters, metrics, Lyapunov, references.

The metrics report is the comparison table; every number in it comes
from the same cached computations that produce the figures.  The
Lyapunov report records the exact phase-contraction check for each case.
"""
import _bootstrap  # noqa: F401

from trisikon import __version__
from trisikon.cases import Config, case_list
from trisikon.dynamics import damped_dissipation, phase_contraction
from trisikon.io_utils import write_csv, write_report
from trisikon.lyapunov import (contraction_residual, kaplan_yorke_dimension,
                               ks_entropy_rate)
from trisikon.pipeline import build

cfg = Config()
res = build(cfg)
rows = res["rows"]
spectra = res["spectra"]
cases = case_list()


def _line(row):
    """Format one metrics row as fixed-width fields."""
    return (f"{row['case']:<2} {row['label']:<17} "
            f"{row['lambda1_breath']:+.4f} "
            f"{row['perm_entropy']:.3f} {row['complexity']:.3f} "
            f"{row['diff_entropy_mean']:+6.2f} "
            f"{row['frac_harmony']:.2f} {row['frac_transition']:.2f} "
            f"{row['frac_chaos']:.2f} {row['symmetry_breaking']:.3f}")


header = ("key label             <lam1>   H_PE  C_JS  Hdiff   fH   fT   "
          "fC   sym")

write_csv("metrics_cases", {
    k: [row[k] for row in rows] for k in rows[0]
})

metrics_block = [header, "-" * len(header)] + [_line(r) for r in rows]
notes = [
    "<lam1> largest Lyapunov exponent averaged over the breathing cycle",
    "H_PE  permutation entropy; C_JS  statistical complexity (realized)",
    "Hdiff mean k-NN differential entropy of the breathing ensemble",
    "fH,fT,fC time fraction in harmony, transition, chaos (shared bounds)",
    "sym   symmetry-breaking order parameter (ensemble axis-energy CV)",
    "per-case spectra, Kaplan-Yorke dimension and the exact sum rule",
    "are in lyapunov.txt; the route to chaos is in fig02.",
]
write_report("metrics", "TRISIKON case comparison",
             [("table", metrics_block), ("columns", notes)])

lyap_block = []
for c in cases:
    spec = spectra[c.key]
    b_vec = damped_dissipation(c.b_chaos, c.damping_ratio)
    exact = phase_contraction(b_vec)
    lyap_block.append(
        f"{c.key} {c.label}: lambda = "
        f"[{spec[0]:+.4f}, {spec[1]:+.4f}, {spec[2]:+.4f}]")
    lyap_block.append(
        f"   D_KY = {kaplan_yorke_dimension(spec):.3f}   "
        f"h_KS = {ks_entropy_rate(spec):.4f}")
    lyap_block.append(
        f"   sum = {spec.sum():+.6f}   exact -(b1+b2+b3) = {exact:+.6f}   "
        f"residual = {contraction_residual(spec, b_vec):+.2e}")
write_report("lyapunov", "Lyapunov spectra and the exact sum rule",
             [("per case at the chaotic-end dissipation", lyap_block)])

params = [
    f"package version: {__version__}",
    f"seed: {cfg.seed}",
    f"integration step dt: {cfg.dt}",
    f"breathing period: {cfg.breath_period_s} s",
    f"realized trajectory: {cfg.realize_steps} steps at dt={cfg.realize_dt}",
    f"embedding: dx={cfg.embed_dx}, tau={cfg.embed_tau}",
    "",
    "cases (b_harmony, b_chaos, damping ratio):",
]
for c in cases:
    params.append(f"  {c.key} {c.label}: {c.b_harmony}, {c.b_chaos}, "
                  f"{c.damping_ratio}")
write_report("parameters", "Parameters", [("configuration", params)])

refs = [
    "Thomas, R. (1999). Int. J. Bifurcation Chaos 9, 1889.",
    "Benettin, G. et al. (1980). Meccanica 15, 9.",
    "Kaplan, J. L. & Yorke, J. A. (1979). LNM 730, 204.",
    "Pesin, Ya. B. (1977). Russ. Math. Surv. 32, 55.",
    "Kozachenko, L. F. & Leonenko, N. N. (1987). Probl. Inf. Transm. 23, 95.",
    "Bandt, C. & Pompe, B. (2002). Phys. Rev. Lett. 88, 174102.",
    "Lamberti, P. W. et al. (2004). Physica A 334, 119.",
    "Rosso, O. A. et al. (2007). Phys. Rev. Lett. 99, 154102.",
    "Pessa, A. A. B. & Ribeiro, H. V. (2021). Chaos 31, 063110 (ordpy).",
]
write_report("references", "References", [("works cited", refs)])

limits = [
    "TRISIKON as a cinematic concept is taken from the user description;",
    "its mapping to the Thomas flow is a formal analogy, not an empirical",
    "claim about film practice.",
    "",
    "Per-case Lyapunov and complexity points are evaluated at the",
    "chaotic-end dissipation of each case, a representative choice; the",
    "full route to chaos is in the dissipation sweep (fig02).",
    "",
    "The Thomas flow is weakly chaotic, so the positive exponent is small;",
    "the sum rule nonetheless pins the full spectrum to machine accuracy.",
    "",
    "Case D breaks the cyclic symmetry and is best read as a boundary of",
    "the concept rather than a member of it.",
]
write_report("open_items", "Open items and limitations",
             [("notes", limits)])
print("reports written")
