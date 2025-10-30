"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { Play, TrendingUp, TrendingDown, Loader2 } from "lucide-react"

interface EvalResults {
  auc: number
  seed_score: number
  panel_mean: number
  panel_max: number
  improvement: number
}

export function EvaluatePanel() {
  const { toast } = useToast()
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [results, setResults] = useState<EvalResults | null>(null)

  useEffect(() => {
    const saved = sessionStorage.getItem("abyss_eval_results")
    if (saved) {
      try {
        setResults(JSON.parse(saved))
      } catch (e) {
        console.error("Failed to load eval results:", e)
      }
    }
  }, [])

  const runEvaluation = async () => {
    setIsEvaluating(true)

    try {
      const config = sessionStorage.getItem("abyss_config")
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: config || "{}",
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.error || "Evaluation failed")

      setResults(data)
      sessionStorage.setItem("abyss_eval_results", JSON.stringify(data))

      toast({
        title: "Evaluation Complete",
        description: `Model AUC: ${data.auc.toFixed(3)}`,
      })
    } catch (error: any) {
      toast({
        title: "Evaluation Failed",
        description: error.message,
        variant: "destructive",
      })
    } finally {
      setIsEvaluating(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Evaluate Performance</h2>
          <p className="text-sm text-muted-foreground mt-1">Assess model performance and panel quality</p>
        </div>
        <Button onClick={runEvaluation} disabled={isEvaluating}>
          {isEvaluating ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Evaluating...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Run Evaluation
            </>
          )}
        </Button>
      </div>

      {!results ? (
        <Card className="border-border/50">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <TrendingUp className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium text-muted-foreground">No evaluation results yet</p>
            <p className="text-sm text-muted-foreground mt-2">Run evaluation to assess model performance</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
              <CardDescription>ROC AUC score on validation set</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">AUC Score</span>
                  <Badge variant={results.auc >= 0.9 ? "default" : "secondary"}>
                    {results.auc >= 0.9 ? "Excellent" : results.auc >= 0.8 ? "Good" : "Fair"}
                  </Badge>
                </div>
                <Progress value={results.auc * 100} className="h-3" />
                <p className="text-2xl font-bold">{results.auc.toFixed(3)}</p>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Seed vs Panel Comparison</CardTitle>
                <CardDescription>Score improvements over baseline</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Seed Score</span>
                    <span className="text-sm font-medium">{results.seed_score.toFixed(3)}</span>
                  </div>
                  <Progress value={results.seed_score * 100} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Panel Mean</span>
                    <span className="text-sm font-medium">{results.panel_mean.toFixed(3)}</span>
                  </div>
                  <Progress value={results.panel_mean * 100} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Panel Max</span>
                    <span className="text-sm font-medium">{results.panel_max.toFixed(3)}</span>
                  </div>
                  <Progress value={results.panel_max * 100} className="h-2" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>Improvement Metrics</CardTitle>
                <CardDescription>Performance gains over baseline</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg bg-card/50 border border-border/50">
                  <div className="flex items-center gap-2">
                    {results.improvement > 0 ? (
                      <TrendingUp className="h-5 w-5 text-green-500" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-red-500" />
                    )}
                    <span className="text-sm font-medium">Overall Improvement</span>
                  </div>
                  <span className="text-2xl font-bold">{results.improvement.toFixed(1)}%</span>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Mean vs Seed</span>
                    <span className="font-medium">{((results.panel_mean - results.seed_score) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Max vs Seed</span>
                    <span className="font-medium">{((results.panel_max - results.seed_score) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
