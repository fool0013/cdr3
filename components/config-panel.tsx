"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { useToast } from "@/hooks/use-toast"
import { Save, RotateCcw } from "lucide-react"

export function ConfigPanel() {
  const { toast } = useToast()
  const [config, setConfig] = useState({
    antigen: "",
    beamWidth: 5,
    topK: 100,
    temperature: 1.0,
    minLength: 8,
    maxLength: 20,
    numClusters: 10,
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001,
  })

  useEffect(() => {
    const saved = sessionStorage.getItem("abyss_config")
    if (saved) {
      try {
        setConfig(JSON.parse(saved))
      } catch (e) {
        console.error("Failed to load config:", e)
      }
    }
  }, [])

  const validateAntigen = (seq: string) => {
    const validAA = /^[ACDEFGHIKLMNPQRSTVWY]*$/
    return validAA.test(seq.toUpperCase())
  }

  const handleSave = () => {
    if (!config.antigen) {
      toast({
        title: "Validation Error",
        description: "Antigen sequence is required",
        variant: "destructive",
      })
      return
    }

    if (!validateAntigen(config.antigen)) {
      toast({
        title: "Validation Error",
        description: "Antigen must contain only valid amino acids (A-Z)",
        variant: "destructive",
      })
      return
    }

    sessionStorage.setItem("abyss_config", JSON.stringify(config))
    toast({
      title: "Configuration Saved",
      description: "Settings have been saved successfully",
    })
  }

  const handleReset = () => {
    const defaults = {
      antigen: "",
      beamWidth: 5,
      topK: 100,
      temperature: 1.0,
      minLength: 8,
      maxLength: 20,
      numClusters: 10,
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
    }
    setConfig(defaults)
    sessionStorage.setItem("abyss_config", JSON.stringify(defaults))
    toast({
      title: "Configuration Reset",
      description: "Settings have been reset to defaults",
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Configuration</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Configure parameters for CDR3 generation and optimization
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleReset}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button onClick={handleSave}>
            <Save className="h-4 w-4 mr-2" />
            Save Configuration
          </Button>
        </div>
      </div>

      <div className="grid gap-6">
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle>Target Antigen</CardTitle>
            <CardDescription>Specify the target antigen sequence</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="antigen">Antigen Sequence</Label>
              <Input
                id="antigen"
                placeholder="Enter amino acid sequence (uppercase)"
                value={config.antigen}
                onChange={(e) => {
                  const val = e.target.value.toUpperCase()
                  setConfig({ ...config, antigen: val })
                }}
                className="font-mono"
              />
              <p className="text-xs text-muted-foreground">
                Valid amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle>Generation Settings</CardTitle>
            <CardDescription>Control the CDR3 generation process</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Beam Width: {config.beamWidth}</Label>
              <Slider
                value={[config.beamWidth]}
                onValueChange={([val]) => setConfig({ ...config, beamWidth: val })}
                min={1}
                max={20}
                step={1}
              />
            </div>
            <div className="space-y-2">
              <Label>Top-K Candidates: {config.topK}</Label>
              <Slider
                value={[config.topK]}
                onValueChange={([val]) => setConfig({ ...config, topK: val })}
                min={10}
                max={500}
                step={10}
              />
            </div>
            <div className="space-y-2">
              <Label>Temperature: {config.temperature.toFixed(2)}</Label>
              <Slider
                value={[config.temperature]}
                onValueChange={([val]) => setConfig({ ...config, temperature: val })}
                min={0.1}
                max={2.0}
                step={0.1}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle>Sequence Constraints</CardTitle>
            <CardDescription>Define CDR3 length boundaries</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="minLength">Min Length</Label>
                <Input
                  id="minLength"
                  type="number"
                  value={config.minLength}
                  onChange={(e) => setConfig({ ...config, minLength: Number.parseInt(e.target.value) || 8 })}
                  min={5}
                  max={15}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxLength">Max Length</Label>
                <Input
                  id="maxLength"
                  type="number"
                  value={config.maxLength}
                  onChange={(e) => setConfig({ ...config, maxLength: Number.parseInt(e.target.value) || 20 })}
                  min={10}
                  max={30}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Number of Clusters: {config.numClusters}</Label>
              <Slider
                value={[config.numClusters]}
                onValueChange={([val]) => setConfig({ ...config, numClusters: val })}
                min={3}
                max={50}
                step={1}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle>Training Settings</CardTitle>
            <CardDescription>Configure model training parameters</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="epochs">Epochs</Label>
                <Input
                  id="epochs"
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: Number.parseInt(e.target.value) || 10 })}
                  min={1}
                  max={100}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="batchSize">Batch Size</Label>
                <Input
                  id="batchSize"
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: Number.parseInt(e.target.value) || 32 })}
                  min={8}
                  max={128}
                  step={8}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="learningRate">Learning Rate</Label>
                <Input
                  id="learningRate"
                  type="number"
                  value={config.learningRate}
                  onChange={(e) => setConfig({ ...config, learningRate: Number.parseFloat(e.target.value) || 0.001 })}
                  min={0.0001}
                  max={0.1}
                  step={0.0001}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
