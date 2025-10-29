import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    let candidates = body.data
    const config = body.config || { keep_top: 50 }

    // If data is an object with candidates property, extract it
    if (candidates && typeof candidates === "object" && !Array.isArray(candidates) && candidates.candidates) {
      candidates = candidates.candidates
    }

    // Validate we have an array of candidates
    if (!candidates || !Array.isArray(candidates) || candidates.length === 0) {
      return NextResponse.json(
        {
          error: "No candidates found. Run optimization first.",
        },
        { status: 400 },
      )
    }

    // Sort by score and keep top N
    const sorted = [...candidates].sort((a: any, b: any) => (b.score || 0) - (a.score || 0))
    const filtered = sorted.slice(0, config.keep_top || 50)

    return NextResponse.json({
      success: true,
      output: `filtered_${Date.now()}.csv`,
      count: filtered.length,
      demo: true,
      data: filtered,
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
