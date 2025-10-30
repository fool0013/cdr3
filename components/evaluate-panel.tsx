"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { BarChart3, Loader2, TrendingUp } from "lucide-react"

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

  const runEvaluation = async () => {
    setIsEvaluating(true)

    try {
      const config = JSON.parse(sessionStorage.getItem("abyss_config") || "{}")

      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      })

      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Evaluation failed")

      setResults(data)
      sessionStorage.setItem("abyss_eval_results", JSON.stringify(data))

      toast({
        title: "Evaluation Complete",
        description: `Model AUC: ${data.auc.toFixed(3)}`,
      })
    } catch (error) {
      toast({
        title: "Evaluation Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setIsEvaluating(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Run Evaluation
          </CardTitle>
          <CardDescription>Evaluate model performance and candidate quality</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={runEvaluation} disabled={isEvaluating} size="lg" className="w-full gap-2">
            {isEvaluating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Evaluating...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4" />
                Run Evaluation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {results && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
              <CardDescription>ROC AUC score on test set</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">AUC Score</span>
                <span className="text-2xl font-bold text-blue-500">{results.auc.toFixed(3)}</span>
              </div>
              <Progress value={results.auc * 100} className="h-2" />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Candidate Improvement
              </CardTitle>
              <CardDescription>Comparison of seed vs optimized panel</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Seed Score</div>
                  <div className="text-2xl font-bold">{results.seed_score.toFixed(3)}</div>
                </div>
                <div className="space-y-2">
                  <div className="text-sm text-muted-foreground">Panel Mean</div>
                  <div className="text-2xl font-bold text-blue-500">{results.panel_mean.toFixed(3)}</div>
                </div>
              </div>

              <div className="pt-4 border-t">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Improvement</span>
                  <span className="text-xl font-bold text-green-500">+{results.improvement.toFixed(1)}%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
