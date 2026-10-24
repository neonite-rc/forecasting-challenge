# IP Protection & Enforcement Playbook

How the repository's provenance markers work, how to detect unauthorized
commercial or production use, and how to turn detection into a takedown. This
is practical guidance, not legal advice — have a qualified lawyer review any
formal notice before sending it.

## 1. What the license forbids

The original solution work is under the **PolyForm Noncommercial License
1.0.0** ([`LICENSE`](../LICENSE)). Review, testing, research, noncommercial
modification and noncommercial sharing are allowed. The following require
prior written authorization and are copyright infringement without it:

- running the software, or a derivative, in a **commercial production**
  environment or internal business operation;
- incorporating it into a proprietary product, service or customer-facing API;
- commercial redistribution or sublicensing.

The top-level LICENSE adds an explicit, narrow grant so the recruitment team
can lawfully run and evaluate the submission — that grant does not extend to
production or commercial use.

## 2. Provenance markers and where they live

| Marker | Location | Purpose |
|---|---|---|
| Required Notice + SPDX id + build tag `CWFC-NC1-7Q2M9X4D` | Header of every proprietary source file | Survives ordinary refactors; removal breaches the PolyForm Notices and Violations clauses and terminates the license |
| `src/cwfc/_about.py` | Author, contact, copyright, build tag, repository | Single canonical metadata module; surfaced by `cwfc.predict --about` |
| `docs/PROVENANCE.json` | SHA-256 of every proprietary file at release | Tamper evidence; regenerate with `python tools/provenance.py generate` |
| Timestamped Git history | GitHub commits under the author identity | Independent, third-party-dated proof of authorship and first publication |
| Distinctive identifiers | `empirical_interval`, `capacity_trigger`, `_cv_window`, `fold_band_breach_rates`, the 10-fold CV constants | Searchable fingerprints of the original methodology |

Markers are deliberately placed only in proprietary source — never in tests,
generated CSVs, or the upstream MIT files — so they cannot change evaluation
results or the submission contract.

## 3. Why there is no hidden beacon, and what to do instead

A covert phone-home routine in a take-home is (a) visible to reviewers and
reads as malware, risking the application; (b) blocked by air-gapped review
environments; (c) potentially unlawful to operate without disclosure, even on
your own code; and (d) unnecessary, because takedowns rest on authorship
evidence, not telemetry. Detection therefore uses passive monitoring:

1. **Code search, run monthly and after any rejection/no-hire outcome:**
   - GitHub code search (logged-in): `CWFC-NC1-7Q2M9X4D`,
     `capacity_trigger`, `fold_band_breach_rates`, `_cv_window`
   - https://grep.app and https://sourcegraph.com/search with the same strings
   - web search for the build tag and the exact `predictions.csv` header
     combined with the company name
2. **Watch the company's public footprint:** engineering blogs, release notes,
   job descriptions, and public API schemas. Wholesale adoption of your exact
   output schema (`timestamp,active_vm_count,aggregate_cpu_utilization`) or
   your distinctive function names is strong circumstantial evidence.
3. **If a fork or public repo appears:** preserve it immediately (clone,
   screenshots, archived URL with a timestamping service) before contacting
   anyone — repositories can be deleted after a complaint.
4. **Verify a recovered copy** against your release manifest:
   `python tools/provenance.py verify`. A `NOTICE_REMOVED` result documents
   willful deletion of the attribution, which strengthens any later claim.

## 4. Evidence to preserve now

- The local repository with full `.git` history and the remote GitHub URL.
- A signed/exported copy of the GitHub commit history (Settings → export, or
  `git log --pretty=fuller` archived with the bundle: `git bundle create`).
- `docs/PROVENANCE.json` regenerated at each release and committed, so hashes
  are themselves timestamped by GitHub.
- The original challenge brief and submission dates, showing first ownership.
- Records of every distribution: to whom, when, and under what terms
  (the evaluation grant in LICENSE names the recruitment context).

## 5. Cease-and-desist and takedown workflow

1. **Identify the infringement** with URLs, repo names, service endpoints,
   and side-by-side matches (marker strings, function names, manifest hashes).
2. **Send a cease-and-desist letter/email** referencing the PolyForm
   Noncommercial terms, the Required Notice, the build tag, and a deadline
   (typically 10–15 days) to stop use and certify deletion. Keep it factual.
3. **Platform notices if they do not comply:**
   - **GitHub:** Report a repository violation → DMCA takedown request.
   - **AWS / GCP / Azure:** report copyright infringement via their abuse or
     legal portals; include the same DMCA elements.
   - **Vercel / Netlify / other hosts:** each publishes a DMCA contact form.
4. **A valid DMCA notice must contain, per 17 U.S.C. § 512(c)(3):**
   - identification of the copyrighted work (this repository, commit hashes);
   - identification and URL of the infringing material;
   - your contact information;
   - a good-faith belief statement that use is not authorized;
   - a statement, under penalty of perjury, that the information is accurate
     and you are authorized to act;
   - physical or electronic signature.
5. **Counter-notice risk:** the other party may file a counter-notice; be
   prepared to file a court action within 10–14 days or the host may restore
   the material. This is the point at which lawyer involvement is essential.

## 6. Jurisdiction notes

- Copyright is automatic upon creation in India (Copyright Act 1957) and in
  Berne Convention countries, including the US; no registration is required to
  own it or send a takedown. US copyright registration, if you expect US
  litigation, unlocks statutory damages and attorney fees and can be filed
  after the infringement occurs (with limitations).
- PolyForm Noncommercial 1.0.0 is a recognized, lawyer-drafted license with a
  stable SPDX identifier (`PolyForm-Noncommercial-1.0.0`), which makes the
  scope of the grant unambiguous in a dispute.
- The upstream MIT-licensed starter files remain MIT (`LICENSE-MIT`); claims
  must target the original work listed in `NOTICE.md`, never the upstream
  material.

## 7. Maintenance rules

- After any code change, rerun `python tools/provenance.py generate` and
  commit the refreshed manifest in the same change.
- Never delete the build tag or header notices, including when AI coding
  assistants "tidy" the code; their retention is what makes removal by others
  a willful license violation.
- Keep the evaluation grant narrow and do not add broader permissions in
  pull requests.
