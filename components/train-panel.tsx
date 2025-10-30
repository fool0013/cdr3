"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Upload, GraduationCap, Loader2 } from "lucide-react"

export function TrainPanel() {
  const { toast } = useToast()
  const [isTraining, setIsTraining] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.name.endsWith(".csv")) {
        toast({
          title: "Invalid File",
          description: "Please upload a CSV file",
          variant: "destructive",
        })
        return
      }
      setUploadedFile(file)
      toast({
        title: "File Uploaded",
        description: `${file.name} is ready for training`,
      })
    }
  }

  const handleTrain = async () => {
    setIsTraining(true)

    try {
      const config = JSON.parse(sessionStorage.getItem("abyss_config") || "{}")

      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      })

      const data = await res.json()

      if (!res.ok) throw new Error(data.error || "Training failed")

      toast({
        title: "Training Complete",
        description: `Model trained with AUC: ${data.auc?.toFixed(3) || "N/A"}`,
      })
    } catch (error) {
      toast({
        title: "Training Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      })
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Training Data
          </CardTitle>
          <CardDescription>Upload a CSV file with antigen-CDR3 pairs and labels</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-muted-foreground/50 transition-colors">
            <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="file-upload" />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm font-medium mb-1">Click to upload or drag and drop</p>
              <p className="text-xs text-muted-foreground">CSV files only</p>
            </label>
          </div>

          {uploadedFile && (
            <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
              <span className="text-sm font-medium">{uploadedFile.name}</span>
              <span className="text-xs text-muted-foreground">{(uploadedFile.size / 1024).toFixed(1)} KB</span>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GraduationCap className="h-5 w-5" />
            Train Model
          </CardTitle>
          <CardDescription>Train the scoring model on your data</CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleTrain} disabled={isTraining} size="lg" className="w-full gap-2">
            {isTraining ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Training Model...
              </>
            ) : (
              <>
                <GraduationCap className="h-4 w-4" />
                Start Training
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
