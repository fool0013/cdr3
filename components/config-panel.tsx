"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { Save } from "lucide-react"

interface Config {
  antigen_sequence: string
  seed_cdr3: string
  num_candidates: number
  beam_width: number
  max_length: number
  top_n: number
  num_clusters: number
  train_data_path: string
  model_path: string
  test_split: number
}

export function ConfigPanel() {
  const { toast } = useToast()
  const [config, setConfig] = useState<Config>({
    antigen_sequence: "",
    seed_cdr3: "",
    num_candidates: 100,
    beam_width: 5,
    max_length: 20,
    top_n: 50,
    num_clusters: 10,
    train_data_path: "data/antigen_cdr3_pairs.csv",
    model_path: "models/scorer.pkl",
    test_split: 0.2,
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

  const validateSequence = (seq: string): boolean => {
    return /^[ACDEFGHIKLMNPQRSTVWY]*$/.test(seq.toUpperCase())
  }

  const handleSave = () => {
    if (config.antigen_sequence && !validateSequence(config.antigen_sequence)) {
      toast({
        title: "Invalid Antigen Sequence",
        description: "Only amino acid letters (A-Z) are allowed",
        variant: "destructive",
      })
      return
    }

    if (config.seed_cdr3 && !validateSequence(config.seed_cdr3)) {
      toast({
        title: "Invalid CDR3 Sequence",
        description: "Only amino acid letters (A-Z) are allowed",
        variant: "destructive",
      })
      return
    }

    sessionStorage.setItem("abyss_config", JSON.stringify(config))
    toast({
      title: "Configuration Saved",
      description: "Your settings have been saved successfully",
    })
  }

  const updateConfig = (key: keyof Config, value: string | number) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Sequence Configuration</CardTitle>
            <CardDescription>Define target antigen and seed CDR3 sequences</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="antigen">Antigen Sequence</Label>
              <Input
                id="antigen"
                placeholder="Enter antigen amino acid sequence (uppercase)"
                value={config.antigen_sequence}
                onChange={(e) => updateConfig("antigen_sequence", e.target.value.toUpperCase())}
                className="font-mono"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="seed">Seed CDR3 Sequence (Optional)</Label>
              <Input
                id="seed"
                placeholder="Enter seed CDR3 sequence (uppercase)"
                value={config.seed_cdr3}
                onChange={(e) => updateConfig("seed_cdr3", e.target.value.toUpperCase())}
                className="font-mono"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation Parameters</CardTitle>
            <CardDescription>Control candidate generation and optimization</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="num_candidates">Number of Candidates</Label>
                <Input
                  id="num_candidates"
                  type="number"
                  min="1"
                  max="1000"
                  value={config.num_candidates}
                  onChange={(e) => updateConfig("num_candidates", Number.parseInt(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="beam_width">Beam Width</Label>
                <Input
                  id="beam_width"
                  type="number"
                  min="1"
                  max="20"
                  value={config.beam_width}
                  onChange={(e) => updateConfig("beam_width", Number.parseInt(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max_length">Max CDR3 Length</Label>
                <Input
                  id="max_length"
                  type="number"
                  min="5"
                  max="30"
                  value={config.max_length}
                  onChange={(e) => updateConfig("max_length", Number.parseInt(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="top_n">Top N to Keep</Label>
                <Input
                  id="top_n"
                  type="number"
                  min="1"
                  max="500"
                  value={config.top_n}
                  onChange={(e) => updateConfig("top_n", Number.parseInt(e.target.value))}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Clustering Settings</CardTitle>
            <CardDescription>Configure diversity clustering parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="num_clusters">Number of Clusters</Label>
              <Input
                id="num_clusters"
                type="number"
                min="1"
                max="50"
                value={config.num_clusters}
                onChange={(e) => updateConfig("num_clusters", Number.parseInt(e.target.value))}
              />
            </div>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Training Settings</CardTitle>
            <CardDescription>Configure model training parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="train_data">Training Data Path</Label>
                <Input
                  id="train_data"
                  value={config.train_data_path}
                  onChange={(e) => updateConfig("train_data_path", e.target.value)}
                  className="font-mono text-sm"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="model_path">Model Save Path</Label>
                <Input
                  id="model_path"
                  value={config.model_path}
                  onChange={(e) => updateConfig("model_path", e.target.value)}
                  className="font-mono text-sm"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="test_split">Test Split Ratio</Label>
                <Input
                  id="test_split"
                  type="number"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={config.test_split}
                  onChange={(e) => updateConfig("test_split", Number.parseFloat(e.target.value))}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-end">
        <Button onClick={handleSave} size="lg" className="gap-2">
          <Save className="h-4 w-4" />
          Save Configuration
        </Button>
      </div>
    </div>
  )
}
