"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Download, FileText, Database } from "lucide-react"

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
    try {
      const data = sessionStorage.getItem("abyss_last_panel")
      if (data) {
        const parsed = JSON.parse(data)
        setCandidates(parsed)

        const total = parsed.length
        const avgScore = parsed.reduce((sum: number, c: Candidate) => sum + c.score, 0) / total
        const clusters = new Set(parsed.map((c: Candidate) => c.cluster)).size

        setStats({ total, avgScore, clusters })
      }
    } catch (e) {
      console.error("Failed to load results:", e)
    }
  }

  const downloadCSV = () => {
    if (candidates.length === 0) {
      toast({
        title: "No Results",
        description: "Run the optimization pipeline first",
        variant: "destructive",
      })
      return
    }

    const csv = ["sequence,score,cluster", ...candidates.map((c) => `${c.sequence},${c.score},${c.cluster || 0}`)].join(
      "\n",
    )

    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "abyss_candidates.csv"
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Download Started",
      description: "Your results are being downloaded",
    })
  }

  const downloadFASTA = () => {
    if (candidates.length === 0) {
      toast({
        title: "No Results",
        description: "Run the optimization pipeline first",
        variant: "destructive",
      })
      return
    }

    const fasta = candidates
      .map((c, i) => `>candidate_${i + 1} score=${c.score.toFixed(3)} cluster=${c.cluster || 0}\n${c.sequence}`)
      .join("\n")

    const blob = new Blob([fasta], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "abyss_candidates.fasta"
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Download Started",
      description: "Your results are being downloaded",
    })
  }

  return (
    <div className="space-y-6">
      {candidates.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Database className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Results Yet</h3>
            <p className="text-sm text-muted-foreground text-center max-w-md">
              Run the optimization pipeline to generate CDR3 candidates. Results will appear here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Total Candidates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-500">{stats.total}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Average Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-500">{stats.avgScore.toFixed(3)}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium">Unique Clusters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-500">{stats.clusters}</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Download CSV
                </CardTitle>
                <CardDescription>Spreadsheet format with scores and clusters</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={downloadCSV} className="w-full gap-2">
                  <Download className="h-4 w-4" />
                  Download CSV
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Download FASTA
                </CardTitle>
                <CardDescription>Sequence format for downstream analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={downloadFASTA} className="w-full gap-2">
                  <Download className="h-4 w-4" />
                  Download FASTA
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Top Candidates Preview</CardTitle>
              <CardDescription>Showing top 10 candidates by score</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {candidates.slice(0, 10).map((c, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                    <span className="font-mono text-sm">{c.sequence}</span>
                    <div className="flex items-center gap-4">
                      <span className="text-sm text-muted-foreground">Cluster {c.cluster || 0}</span>
                      <span className="font-semibold text-blue-500">{c.score.toFixed(3)}</span>
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
