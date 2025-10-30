import { NextResponse } from "next/server"

function simpleKMeans(data: number[][], k: number, maxIter = 10): number[] {
  const n = data.length
  const dim = data[0].length

  // Initialize centroids randomly
  const centroids: number[][] = []
  const indices = new Set<number>()
  while (centroids.length < k) {
    const idx = Math.floor(Math.random() * n)
    if (!indices.has(idx)) {
      indices.add(idx)
      centroids.push([...data[idx]])
    }
  }

  const labels = new Array(n).fill(0)

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign points to nearest centroid
    for (let i = 0; i < n; i++) {
      let minDist = Number.POSITIVE_INFINITY
      let bestCluster = 0
      for (let c = 0; c < k; c++) {
        let dist = 0
        for (let d = 0; d < dim; d++) {
          const diff = data[i][d] - centroids[c][d]
          dist += diff * diff
        }
        if (dist < minDist) {
          minDist = dist
          bestCluster = c
        }
      }
      labels[i] = bestCluster
    }

    // Update centroids
    const counts = new Array(k).fill(0)
    const newCentroids = Array(k)
      .fill(0)
      .map(() => new Array(dim).fill(0))

    for (let i = 0; i < n; i++) {
      const c = labels[i]
      counts[c]++
      for (let d = 0; d < dim; d++) {
        newCentroids[c][d] += data[i][d]
      }
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] = newCentroids[c][d] / counts[c]
        }
      }
    }
  }

  return labels
}

function mockEmbedding(seq: string): number[] {
  // Create a simple mock embedding based on sequence properties
  const embedding = new Array(64).fill(0)

  // Encode length
  embedding[0] = seq.length / 20

  // Encode amino acid composition
  for (let i = 0; i < seq.length; i++) {
    const code = seq.charCodeAt(i) - 65 // A=0, B=1, etc.
    const idx = (code % 60) + 1
    embedding[idx] += 1 / seq.length
  }

  // Add some noise for diversity
  for (let i = 0; i < embedding.length; i++) {
    embedding[i] += (Math.random() - 0.5) * 0.1
  }

  return embedding
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const filteredData = body.data
    const config = body.config || {}

    if (!filteredData || !filteredData.candidates) {
      return NextResponse.json(
        {
          error: "No filtered candidates found. Run filtering first.",
        },
        { status: 400 },
      )
    }

    const k = Math.min(config.clusters || 10, filteredData.candidates.length)

    console.log("[v0] Clustering into", k, "clusters")

    // Only cluster passing candidates
    const passingCandidates = filteredData.candidates.filter((c: any) => c.passes === "Y")

    if (passingCandidates.length === 0) {
      return NextResponse.json(
        {
          error: "No passing candidates to cluster",
        },
        { status: 400 },
      )
    }

    // Generate mock embeddings for each sequence
    const embeddings = passingCandidates.map((c: any) => mockEmbedding(c.cdr3))

    // Perform k-means clustering
    const actualK = Math.min(k, passingCandidates.length)
    const labels = simpleKMeans(embeddings, actualK)

    // Select highest-scoring member from each cluster
    const clusterReps: any[] = []
    for (let c = 0; c < actualK; c++) {
      const clusterMembers = passingCandidates
        .map((candidate: any, idx: number) => ({ candidate, idx }))
        .filter(({ idx }) => labels[idx] === c)

      if (clusterMembers.length > 0) {
        const best = clusterMembers.reduce((a, b) => (a.candidate.score > b.candidate.score ? a : b))
        clusterReps.push({
          ...best.candidate,
          cluster: c,
        })
      }
    }

    // Sort by score descending
    clusterReps.sort((a, b) => b.score - a.score)

    const resultData = {
      timestamp: filteredData.timestamp,
      antigen: filteredData.antigen,
      candidates: clusterReps,
      count: clusterReps.length,
    }

    console.log("[v0] Clustered:", { clusters: actualK, representatives: clusterReps.length })

    return NextResponse.json({
      success: true,
      output: `opt_${filteredData.timestamp}_panel.csv`,
      count: clusterReps.length,
      data: resultData,
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
