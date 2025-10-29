import { NextResponse } from "next/server"

export async function POST() {
  try {
    const rawDataStr = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("abyss_last_raw") : null

    if (!rawDataStr) {
      return NextResponse.json(
        {
          error: "No raw candidates found. Run optimization first.",
        },
        { status: 400 },
      )
    }

    const rawData = JSON.parse(rawDataStr)
    const configStr = typeof localStorage !== "undefined" ? localStorage.getItem("abyss_config") : null
    const config = configStr ? JSON.parse(configStr) : { keep_top: 50 }

    const sorted = [...rawData.candidates].sort((a, b) => b.score - a.score)
    const filtered = sorted.slice(0, config.keep_top || 50)

    const resultData = {
      timestamp: rawData.timestamp,
      antigen: rawData.antigen,
      candidates: filtered,
      count: filtered.length,
    }

    // Store for next step
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem("abyss_last_filtered", JSON.stringify(resultData))
    }

    return NextResponse.json({
      success: true,
      output: `opt_${rawData.timestamp}_filtered.csv`,
      count: filtered.length,
      demo: true,
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
