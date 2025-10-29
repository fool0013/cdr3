"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Loader2, Play } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface EvalResults {
  auc?: number
  seed_score?: number
  panel_mean?: number
  panel_max?: number
  improvement?: number
}

export function EvaluatePanel() {
  const { toast } = useToast()
  const [evaluating, setEvaluating] = useState(false)
  const [results, setResults] = useState<EvalResults | null>(null)

  const runEvaluation = async () => {
    setEvaluating(true)
    setResults(null)

    try {
      const res = await fetch("/api/evaluate", { method: "POST" })
      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Evaluation failed")

      setResults(data)

      if (typeof sessionStorage !== "undefined") {
        sessionStorage.setItem("abyss_eval_results", JSON.stringify(data))
      }

      toast({
        title: "Success",
        description: "Evaluation completed successfully",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to run evaluation",
        variant: "destructive",
      })
    } finally {
      setEvaluating(false)
    }
  }

  useEffect(() => {
    if (typeof sessionStorage !== "undefined") {
      const stored = sessionStorage.getItem("abyss_eval_results")
      if (stored) {
        setResults(JSON.parse(stored))
      }
    }
  }, [])

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Model Evaluation</CardTitle>
          <CardDescription>Evaluate model performance and compare seed vs panel</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription>
              Evaluation includes global AUC on test data and comparison between seed CDR3 and generated panel.
            </AlertDescription>
          </Alert>

          <Button onClick={runEvaluation} disabled={evaluating} className="w-full" size="lg">
            {evaluating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Evaluating...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Evaluation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <>
          {results.auc !== undefined && (
            <Card>
              <CardHeader>
                <CardTitle>Global Performance</CardTitle>
                <CardDescription>Model AUC on test data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">{results.auc.toFixed(4)}</div>
                  <p className="mt-2 text-sm text-muted-foreground">Area Under Curve (AUC)</p>
                </div>
              </CardContent>
            </Card>
          )}

          {results.seed_score !== undefined && (
            <Card>
              <CardHeader>
                <CardTitle>Seed vs Panel</CardTitle>
                <CardDescription>Comparison of binding scores</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border border-border p-4">
                  <span className="text-sm font-medium">Seed CDR3 Score</span>
                  <span className="text-lg font-bold">{results.seed_score.toFixed(4)}</span>
                </div>

                {results.panel_mean !== undefined && (
                  <div className="flex items-center justify-between rounded-lg border border-border p-4">
                    <span className="text-sm font-medium">Panel Mean Score</span>
                    <span className="text-lg font-bold">{results.panel_mean.toFixed(4)}</span>
                  </div>
                )}

                {results.panel_max !== undefined && (
                  <div className="flex items-center justify-between rounded-lg border border-border p-4">
                    <span className="text-sm font-medium">Panel Max Score</span>
                    <span className="text-lg font-bold text-primary">{results.panel_max.toFixed(4)}</span>
                  </div>
                )}

                {results.improvement !== undefined && (
                  <Alert>
                    <AlertDescription>
                      <span className="font-medium">Improvement: </span>
                      <span className={results.improvement > 0 ? "text-green-500" : "text-red-500"}>
                        {results.improvement > 0 ? "+" : ""}
                        {(results.improvement * 100).toFixed(2)}%
                      </span>
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
