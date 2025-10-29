import { NextResponse } from "next/server"

function clusterCandidates(candidates: any[], k: number) {
  // Sort by score and take top from each score range
  const sorted = [...candidates].sort((a, b) => b.score - a.score)
  const clustered = []
  const step = Math.floor(sorted.length / k)

  for (let i = 0; i < k && i * step < sorted.length; i++) {
    clustered.push({
      ...sorted[i * step],
      cluster: i,
    })
  }

  return clustered
}

export async function POST() {
  try {
    const filteredDataStr = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("abyss_last_filtered") : null

    if (!filteredDataStr) {
      return NextResponse.json(
        {
          error: "No filtered candidates found. Run filtering first.",
        },
        { status: 400 },
      )
    }

    const filteredData = JSON.parse(filteredDataStr)
    const configStr = typeof localStorage !== "undefined" ? localStorage.getItem("abyss_config") : null
    const config = configStr ? JSON.parse(configStr) : { clusters: 10 }

    const clustered = clusterCandidates(filteredData.candidates, config.clusters || 10)

    const resultData = {
      timestamp: filteredData.timestamp,
      antigen: filteredData.antigen,
      candidates: clustered,
      count: clustered.length,
    }

    // Store final results
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem("abyss_last_panel", JSON.stringify(resultData))
    }

    return NextResponse.json({
      success: true,
      output: `opt_${filteredData.timestamp}_panel.csv`,
      count: clustered.length,
      demo: true,
    })
  } catch (error) {
    console.error("[v0] Cluster error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Clustering failed",
      },
      { status: 500 },
    )
  }
}
