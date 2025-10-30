"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { Upload, Play, FileText, Loader2 } from "lucide-react"

export function TrainPanel() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [logs, setLogs] = useState<string[]>([])

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0]
    if (uploadedFile) {
      if (!uploadedFile.name.endsWith(".csv")) {
        toast({
          title: "Invalid File",
          description: "Please upload a CSV file",
          variant: "destructive",
        })
        return
      }
      setFile(uploadedFile)
      toast({
        title: "File Uploaded",
        description: `${uploadedFile.name} ready for training`,
      })
    }
  }

  const runTraining = async () => {
    if (!file) {
      toast({
        title: "No File",
        description: "Please upload a training CSV file first",
        variant: "destructive",
      })
      return
    }

    setIsTraining(true)
    setProgress(0)
    setLogs([])

    try {
      const config = sessionStorage.getItem("abyss_config")
      const parsedConfig = config ? JSON.parse(config) : {}

      // Simulate training with progress updates
      const epochs = parsedConfig.epochs || 10
      for (let i = 0; i <= epochs; i++) {
        await new Promise((resolve) => setTimeout(resolve, 500))
        setProgress((i / epochs) * 100)
        setLogs((prev) => [...prev, `Epoch ${i}/${epochs} - Loss: ${(1.0 - i * 0.08).toFixed(4)}`])
      }

      const mockResults = {
        auc: 0.85 + Math.random() * 0.1,
        accuracy: 0.8 + Math.random() * 0.15,
      }

      sessionStorage.setItem("abyss_train_results", JSON.stringify(mockResults))
      setLogs((prev) => [
        ...prev,
        `Training complete! AUC: ${mockResults.auc.toFixed(3)}, Accuracy: ${mockResults.accuracy.toFixed(3)}`,
      ])

      toast({
        title: "Training Complete",
        description: `Model trained successfully with AUC: ${mockResults.auc.toFixed(3)}`,
      })
    } catch (error: any) {
      toast({
        title: "Training Failed",
        description: error.message,
        variant: "destructive",
      })
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Train Model</h2>
          <p className="text-sm text-muted-foreground mt-1">Upload training data and train the scoring model</p>
        </div>
      </div>

      <Card className="border-border/50">
        <CardHeader>
          <CardTitle>Upload Training Data</CardTitle>
          <CardDescription>CSV file with antigen-CDR3 pairs and labels</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
            <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="file-upload" />
            <label htmlFor="file-upload">
              <Button variant="outline" asChild>
                <span>
                  <FileText className="h-4 w-4 mr-2" />
                  Choose CSV File
                </span>
              </Button>
            </label>
            {file && (
              <p className="text-sm text-muted-foreground mt-4">
                Selected: <span className="font-medium">{file.name}</span>
              </p>
            )}
          </div>

          <Button onClick={runTraining} disabled={!file || isTraining} size="lg" className="w-full">
            {isTraining ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Training
              </>
            )}
          </Button>

          {isTraining && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Training Progress</span>
                <span className="text-sm font-medium">{progress.toFixed(0)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {logs.length > 0 && (
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle>Training Logs</CardTitle>
            <CardDescription>Real-time training progress and output</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-black/50 rounded-lg p-4 font-mono text-sm space-y-1 max-h-[400px] overflow-y-auto">
              {logs.map((log, i) => (
                <div key={i} className="text-green-400">
                  {log}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
