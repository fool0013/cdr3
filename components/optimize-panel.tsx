"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { Zap, Filter, Grid3x3, Play, Loader2 } from "lucide-react"

export function OptimizePanel() {
  const { toast } = useToast()
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState<string>("")
  const [stats, setStats] = useState<{ generated: number; filtered: number; clustered: number }>({
    generated: 0,
    filtered: 0,
    clustered: 0,
  })

  const runQuickPipeline = async () => {
    setIsRunning(true)
    setProgress(0)
    setCurrentStep("Generating candidates...")

    try {
      const config = JSON.parse(sessionStorage.getItem("abyss_config") || "{}")

      if (!config.antigen_sequence) {
        toast({
          title: "Missing Configuration",
          description: "Please configure an antigen sequence first",
          variant: "destructive",
        })
        setIsRunning(false)
        return
      }

      // Step 1: Generate
      setProgress(10)
      const genRes = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      })
      const genData = await genRes.json()

      if (!genRes.ok) throw new Error(genData.error || "Failed to generate")

      setStats((prev) => ({ ...prev, generated: genData.candidates?.length || 0 }))
      sessionStorage.setItem("abyss_last_raw", JSON.stringify(genData.candidates || []))
      setProgress(40)

      // Step 2: Filter
      setCurrentStep("Filtering candidates...")
      const filterRes = await fetch("/api/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: genData.candidates, config }),
      })
      const filterData = await filterRes.json()

      if (!filterRes.ok) throw new Error(filterData.error || "Failed to filter")

      setStats((prev) => ({ ...prev, filtered: filterData.candidates?.length || 0 }))
      sessionStorage.setItem("abyss_last_filtered", JSON.stringify(filterData.candidates || []))
      setProgress(70)

      // Step 3: Cluster
      setCurrentStep("Clustering candidates...")
      const clusterRes = await fetch("/api/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: filterData.candidates, config }),
      })
      const clusterData = await clusterRes.json()

      if (!clusterRes.ok) throw new Error(clusterData.error || "Failed to cluster")

      setStats((prev) => ({ ...prev, clustered: clusterData.candidates?.length || 0 }))
      sessionStorage.setItem("abyss_last_panel", JSON.stringify(clusterData.candidates || []))
      setProgress(100)
      setCurrentStep("Pipeline complete!")

      toast({
        title: "Pipeline Complete",
        description: `Generated ${clusterData.candidates?.length || 0} diverse candidates`,
      })
    } catch (error) {
      toast({
        title: "Pipeline Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card className="border-blue-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-500" />
            Quick Pipeline
          </CardTitle>
          <CardDescription>Run the complete optimization pipeline in one click</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button onClick={runQuickPipeline} disabled={isRunning} size="lg" className="w-full gap-2">
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running Pipeline...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Run Quick Pipeline
              </>
            )}
          </Button>

          {isRunning && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{currentStep}</span>
                <span className="font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}

          {stats.generated > 0 && (
            <div className="grid grid-cols-3 gap-4 pt-4 border-t">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-500">{stats.generated}</div>
                <div className="text-xs text-muted-foreground">Generated</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-500">{stats.filtered}</div>
                <div className="text-xs text-muted-foreground">Filtered</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-500">{stats.clustered}</div>
                <div className="text-xs text-muted-foreground">Clustered</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Play className="h-4 w-4" />
              Step 1: Generate
            </CardTitle>
            <CardDescription>Create candidate CDR3 sequences using beam search</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Uses ESM-2 embeddings and learned scoring to generate diverse candidates
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Filter className="h-4 w-4" />
              Step 2: Filter
            </CardTitle>
            <CardDescription>Select top-scoring candidates</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Ranks candidates by predicted binding affinity and keeps the best
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Grid3x3 className="h-4 w-4" />
              Step 3: Cluster
            </CardTitle>
            <CardDescription>Ensure sequence diversity</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Groups similar sequences and selects representatives for diversity
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
