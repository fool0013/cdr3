import { NextResponse } from "next/server"

const AA = "ACDEFGHIKLMNPQRSTVWY"

function mutate(seq: string, k = 1): string {
  const s = seq.split("")
  const L = s.length
  for (let i = 0; i < k; i++) {
    const idx = Math.floor(Math.random() * (L - 1)) + 1 // Never mutate first position
    s[idx] = AA[Math.floor(Math.random() * AA.length)]
  }
  s[0] = "C" // Enforce leading C
  return s.join("")
}

function generateCandidates(
  antigen: string,
  seedCdr3: string,
  steps: number,
  beam: number,
  kMut: number,
): Array<{ cdr3: string; score: number }> {
  // Enforce leading C on seed
  const seed = seedCdr3.startsWith("C") ? seedCdr3 : "C" + seedCdr3.slice(1)

  let pool = [seed]
  let poolScores = [0.5 + Math.random() * 0.3] // Initial score 0.5-0.8

  for (let step = 0; step < steps; step++) {
    const candidates = new Set(pool)

    // Generate mutations from current pool
    for (const seq of pool) {
      for (let i = 0; i < 3; i++) {
        candidates.add(mutate(seq, kMut))
      }
    }

    // Score all candidates (mock scoring based on sequence properties)
    const candidateList = Array.from(candidates)
    const scores = candidateList.map((seq) => {
      // Mock score based on length, charge, hydrophobicity
      let score = 0.5 + Math.random() * 0.3
      // Prefer moderate lengths (12-16)
      const lenPenalty = Math.abs(seq.length - 14) * 0.02
      score -= lenPenalty
      // Slight bonus for diversity
      const uniqueAA = new Set(seq).size
      score += uniqueAA * 0.01
      return Math.max(0.1, Math.min(1.0, score))
    })

    // Keep top beam candidates
    const indices = scores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, beam)

    pool = indices.map((item) => candidateList[item.idx])
    poolScores = indices.map((item) => item.score)
  }

  // Return all candidates from final pool sorted by score
  return pool.map((cdr3, idx) => ({ cdr3, score: poolScores[idx] })).sort((a, b) => b.score - a.score)
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const config = body.config || {}

    const antigen = config.antigen || "EXAMPLE"
    const seedCdr3 = config.seed_cdr3 || "CARDGYW"
    const steps = config.steps || 50
    const beam = config.beam || 100
    const kMut = config.k_mut || 3

    if (!antigen) {
      return NextResponse.json({ error: "No antigen sequence configured" }, { status: 400 })
    }

    console.log("[v0] Generating candidates with:", { antigen, seedCdr3, steps, beam, kMut })

    const candidates = generateCandidates(antigen, seedCdr3, steps, beam, kMut)

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)

    return NextResponse.json({
      success: true,
      output: `opt_${timestamp}.csv`,
      count: candidates.length,
      data: {
        timestamp,
        antigen,
        candidates: candidates.map((c, idx) => ({
          ...c,
          step: idx,
        })),
        count: candidates.length,
      },
    })
  } catch (error) {
    console.error("[v0] Optimize error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Optimization failed",
      },
      { status: 500 },
    )
  }
}
