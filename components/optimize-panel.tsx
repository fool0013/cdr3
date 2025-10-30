"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { Play, Zap, Filter, Grid3x3, CheckCircle2, Loader2 } from "lucide-react"

export function OptimizePanel() {
  const { toast } = useToast()
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState<string>("")
  const [stats, setStats] = useState({ raw: 0, filtered: 0, clustered: 0 })
  const [stepStatus, setStepStatus] = useState({
    generate: "pending",
    filter: "pending",
    cluster: "pending",
  })

  const runQuickPipeline = async () => {
    const config = sessionStorage.getItem("abyss_config")
    if (!config) {
      toast({
        title: "Configuration Required",
        description: "Please configure settings in the Config tab first",
        variant: "destructive",
      })
      return
    }

    setIsRunning(true)
    setProgress(0)
    setStepStatus({ generate: "running", filter: "pending", cluster: "pending" })

    try {
      // Step 1: Generate
      setCurrentStep("Generating CDR3 candidates...")
      setProgress(10)
      const genRes = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: config,
      })
      const genData = await genRes.json()

      if (!genRes.ok) throw new Error(genData.error || "Generation failed")

      setStats((prev) => ({ ...prev, raw: genData.candidates?.length || 0 }))
      setStepStatus({ generate: "complete", filter: "running", cluster: "pending" })
      setProgress(40)
      sessionStorage.setItem("abyss_last_raw", JSON.stringify(genData.candidates || []))

      // Step 2: Filter
      setCurrentStep("Filtering top candidates...")
      const filterRes = await fetch("/api/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidates: genData.candidates, config: JSON.parse(config) }),
      })
      const filterData = await filterRes.json()

      if (!filterRes.ok) throw new Error(filterData.error || "Filtering failed")

      setStats((prev) => ({ ...prev, filtered: filterData.candidates?.length || 0 }))
      setStepStatus({ generate: "complete", filter: "complete", cluster: "running" })
      setProgress(70)
      sessionStorage.setItem("abyss_last_filtered", JSON.stringify(filterData.candidates || []))

      // Step 3: Cluster
      setCurrentStep("Clustering for diversity...")
      const clusterRes = await fetch("/api/cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidates: filterData.candidates, config: JSON.parse(config) }),
      })
      const clusterData = await clusterRes.json()

      if (!clusterRes.ok) throw new Error(clusterData.error || "Clustering failed")

      setStats((prev) => ({ ...prev, clustered: clusterData.candidates?.length || 0 }))
      setStepStatus({ generate: "complete", filter: "complete", cluster: "complete" })
      setProgress(100)
      sessionStorage.setItem("abyss_last_panel", JSON.stringify(clusterData.candidates || []))

      setCurrentStep("Pipeline complete!")
      toast({
        title: "Pipeline Complete",
        description: `Generated ${clusterData.candidates?.length || 0} diverse CDR3 candidates`,
      })
    } catch (error: any) {
      toast({
        title: "Pipeline Failed",
        description: error.message,
        variant: "destructive",
      })
      setStepStatus((prev) => ({
        ...prev,
        [currentStep.includes("Generating") ? "generate" : currentStep.includes("Filtering") ? "filter" : "cluster"]:
          "error",
      }))
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Optimize CDR3 Sequences</h2>
          <p className="text-sm text-muted-foreground mt-1">Generate and optimize antigen-specific CDR3 candidates</p>
        </div>
      </div>

      <Card className="border-blue-500/20 bg-blue-500/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-500" />
            Quick Pipeline
          </CardTitle>
          <CardDescription>Run the complete optimization pipeline</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <Button onClick={runQuickPipeline} disabled={isRunning} size="lg" className="min-w-[200px]">
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run Quick Pipeline
                </>
              )}
            </Button>
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">{currentStep || "Ready to start"}</span>
                <span className="text-sm font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 pt-4 border-t">
            <div className="flex items-center gap-3">
              {stepStatus.generate === "complete" ? (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              ) : stepStatus.generate === "running" ? (
                <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
              ) : (
                <div className="h-5 w-5 rounded-full border-2 border-muted" />
              )}
              <div>
                <p className="text-sm font-medium">Generate</p>
                <p className="text-xs text-muted-foreground">{stats.raw} candidates</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {stepStatus.filter === "complete" ? (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              ) : stepStatus.filter === "running" ? (
                <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
              ) : (
                <div className="h-5 w-5 rounded-full border-2 border-muted" />
              )}
              <div>
                <p className="text-sm font-medium">Filter</p>
                <p className="text-xs text-muted-foreground">{stats.filtered} selected</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {stepStatus.cluster === "complete" ? (
                <CheckCircle2 className="h-5 w-5 text-green-500" />
              ) : stepStatus.cluster === "running" ? (
                <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
              ) : (
                <div className="h-5 w-5 rounded-full border-2 border-muted" />
              )}
              <div>
                <p className="text-sm font-medium">Cluster</p>
                <p className="text-xs text-muted-foreground">{stats.clustered} diverse</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Play className="h-4 w-4" />
              Step 1: Generate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Use beam search to generate candidate CDR3 sequences optimized for the target antigen
            </p>
            <Badge variant="outline">Beam Search</Badge>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Filter className="h-4 w-4" />
              Step 2: Filter
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Score and rank candidates, selecting the top performers based on binding affinity
            </p>
            <Badge variant="outline">ML Scoring</Badge>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Grid3x3 className="h-4 w-4" />
              Step 3: Cluster
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Group similar sequences and select diverse representatives for experimental validation
            </p>
            <Badge variant="outline">Diversity</Badge>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
