"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { Download, FileText, Database, TrendingUp } from "lucide-react"

interface Candidate {
  sequence: string
  score: number
  cluster?: number
}

export function ResultsPanel() {
  const { toast } = useToast()
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [stats, setStats] = useState({ total: 0, avgScore: 0, clusters: 0 })

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = () => {
    const data = sessionStorage.getItem("abyss_last_panel")
    if (data) {
      try {
        const parsed = JSON.parse(data)
        setCandidates(parsed)
        const avg = parsed.reduce((sum: number, c: Candidate) => sum + c.score, 0) / parsed.length
        const uniqueClusters = new Set(parsed.map((c: Candidate) => c.cluster)).size
        setStats({ total: parsed.length, avgScore: avg, clusters: uniqueClusters })
      } catch (e) {
        console.error("Failed to load results:", e)
      }
    }
  }

  const downloadCSV = () => {
    if (candidates.length === 0) {
      toast({
        title: "No Data",
        description: "No candidates to download. Run optimization first.",
        variant: "destructive",
      })
      return
    }

    const csv = [
      "Sequence,Score,Cluster",
      ...candidates.map((c) => `${c.sequence},${c.score.toFixed(4)},${c.cluster || 0}`),
    ].join("\n")

    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `abyss_candidates_${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Download Started",
      description: "CSV file is being downloaded",
    })
  }

  const downloadFASTA = () => {
    if (candidates.length === 0) {
      toast({
        title: "No Data",
        description: "No candidates to download. Run optimization first.",
        variant: "destructive",
      })
      return
    }

    const fasta = candidates
      .map((c, i) => `>Candidate_${i + 1}|Score:${c.score.toFixed(4)}|Cluster:${c.cluster || 0}\n${c.sequence}`)
      .join("\n")

    const blob = new Blob([fasta], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `abyss_candidates_${Date.now()}.fasta`
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Download Started",
      description: "FASTA file is being downloaded",
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Results</h2>
          <p className="text-sm text-muted-foreground mt-1">View and download optimized CDR3 candidates</p>
        </div>
      </div>

      {candidates.length === 0 ? (
        <Card className="border-border/50">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Database className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium text-muted-foreground">No results yet</p>
            <p className="text-sm text-muted-foreground mt-2">Run the optimization pipeline to generate candidates</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card className="border-border/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground">Total Candidates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.total}</div>
              </CardContent>
            </Card>
            <Card className="border-border/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground">Average Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.avgScore.toFixed(3)}</div>
              </CardContent>
            </Card>
            <Card className="border-border/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground">Unique Clusters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stats.clusters}</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Download CSV
                </CardTitle>
                <CardDescription>Tabular format with scores and cluster assignments</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={downloadCSV} className="w-full">
                  <Download className="h-4 w-4 mr-2" />
                  Download CSV
                </Button>
              </CardContent>
            </Card>

            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Download FASTA
                </CardTitle>
                <CardDescription>Sequence format for downstream analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={downloadFASTA} variant="outline" className="w-full bg-transparent">
                  <Download className="h-4 w-4 mr-2" />
                  Download FASTA
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card className="border-border/50">
            <CardHeader>
              <CardTitle>Top Candidates Preview</CardTitle>
              <CardDescription>Showing top 10 candidates by score</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {candidates.slice(0, 10).map((c, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg border border-border/50 bg-card/50"
                  >
                    <div className="flex items-center gap-4">
                      <Badge variant="outline" className="font-mono">
                        #{i + 1}
                      </Badge>
                      <code className="text-sm font-mono">{c.sequence}</code>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant="secondary">Cluster {c.cluster || 0}</Badge>
                      <div className="flex items-center gap-1">
                        <TrendingUp className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium">{c.score.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
