import { NextResponse } from "next/server"

const AA = new Set("ACDEFGHIKLMNPQRSTVWY")
const HYDRO = new Set("VILFWY")

function hasOnlyStdAA(seq: string): boolean {
  return seq.split("").every((ch) => AA.has(ch))
}

function hasNGlyc(seq: string): boolean {
  // N-X-[S/T] where X != P
  for (let i = 0; i < seq.length - 2; i++) {
    if (seq[i] === "N" && seq[i + 1] !== "P" && (seq[i + 2] === "S" || seq[i + 2] === "T")) {
      return true
    }
  }
  return false
}

function maxHydrophobicRun(seq: string): number {
  let run = 0
  let best = 0
  for (const ch of seq) {
    if (HYDRO.has(ch)) {
      run++
      if (run > best) best = run
    } else {
      run = 0
    }
  }
  return best
}

function netCharge(seq: string): number {
  let charge = 0
  for (const ch of seq) {
    if (ch === "K" || ch === "R") charge += 1
    if (ch === "H") charge += 0.1
    if (ch === "D" || ch === "E") charge -= 1
  }
  return charge
}

function hardFailReason(
  seq: string,
  maxCys: number,
  minLen: number,
  maxLen: number,
  maxHydroRun: number,
  maxAbsCharge: number,
): string {
  if (!hasOnlyStdAA(seq)) return "non_std_aa"
  if ((seq.match(/C/g) || []).length > maxCys) return "too_many_cys"
  if (hasNGlyc(seq)) return "NXS_T_motif"
  if (seq.length < minLen || seq.length > maxLen) return "len_out_of_range"
  if (maxHydrophobicRun(seq) >= maxHydroRun) return `hydrophobic_run>=${maxHydroRun}`
  if (Math.abs(netCharge(seq)) > maxAbsCharge) return "charge_out_of_range"
  return ""
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const rawData = body.data
    const config = body.config || {}

    if (!rawData || !rawData.candidates) {
      return NextResponse.json(
        {
          error: "No raw candidates found. Run optimization first.",
        },
        { status: 400 },
      )
    }

    const maxCys = config.max_cys || 2
    const minLen = config.min_len || 9
    const maxLen = config.max_len || 22
    const maxHydroRun = config.max_hydro_run || 6
    const maxAbsCharge = config.max_abs_charge || 12
    const keepTop = config.keep_top || 50

    console.log("[v0] Filtering with:", { maxCys, minLen, maxLen, maxHydroRun, maxAbsCharge, keepTop })

    // Annotate each candidate with filter results
    const annotated = rawData.candidates.map((c: any) => {
      const seq = c.cdr3
      const reason = hardFailReason(seq, maxCys, minLen, maxLen, maxHydroRun, maxAbsCharge)
      return {
        ...c,
        length: seq.length,
        charge: netCharge(seq).toFixed(1),
        hydro_run: maxHydrophobicRun(seq),
        has_nglyc: hasNGlyc(seq),
        warn_no_leading_C: seq[0] !== "C",
        warn_len: seq.length < minLen || seq.length > maxLen,
        passes: reason === "" ? "Y" : "N",
        fail_reason: reason,
      }
    })

    // Sort: passing first, then by score descending
    annotated.sort((a: any, b: any) => {
      if (a.passes !== b.passes) return a.passes === "Y" ? -1 : 1
      return b.score - a.score
    })

    // Keep top K passing sequences
    const passing = annotated.filter((c: any) => c.passes === "Y").slice(0, keepTop)
    const failing = annotated.filter((c: any) => c.passes !== "Y")
    const filtered = [...passing, ...failing]

    const resultData = {
      timestamp: rawData.timestamp,
      antigen: rawData.antigen,
      candidates: filtered,
      count: passing.length,
    }

    console.log("[v0] Filtered:", { total: filtered.length, passing: passing.length, failing: failing.length })

    return NextResponse.json({
      success: true,
      output: `opt_${rawData.timestamp}_filtered.csv`,
      count: passing.length,
      data: resultData,
    })
  } catch (error) {
    console.error("[v0] Filter error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Filtering failed",
      },
      { status: 500 },
    )
  }
}
