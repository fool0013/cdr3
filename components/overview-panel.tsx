"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Dna, Sparkles, Zap, Database, Settings, FileText, GraduationCap } from "lucide-react"

interface OverviewPanelProps {
  onNavigate: (tab: string) => void
}

export function OverviewPanel({ onNavigate }: OverviewPanelProps) {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <Card className="border-2">
        <CardHeader className="text-center pb-4">
          <CardTitle className="text-3xl md:text-4xl font-bold">Welcome to ABYSS</CardTitle>
          <CardDescription className="text-base md:text-lg mt-2">
            De novo design of antigen-specific CDR3 sequences using machine learning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground text-center max-w-3xl mx-auto">
            ABYSS leverages ESM-2 protein language model embeddings and iterative beam search to generate high-affinity
            CDR3 sequences tailored to your target antigen.
          </p>
        </CardContent>
      </Card>

      {/* What is CDR3 Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Dna className="h-5 w-5" />
            Understanding CDR3
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm md:text-base">
          <p className="text-muted-foreground">
            <strong className="text-foreground">Complementarity-Determining Region 3 (CDR3)</strong> is the most
            variable region of antibody heavy and light chains, located at the center of the antigen-binding site.
          </p>
          <p className="text-muted-foreground">
            CDR3 determines antigen specificity and binding affinity, making it the primary target for antibody
            engineering and therapeutic development.
          </p>
        </CardContent>
      </Card>

      {/* The Challenge Section */}
      <Card>
        <CardHeader>
          <CardTitle>The Antibody Discovery Challenge</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm md:text-base">
          <ul className="space-y-2 text-muted-foreground">
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>
                <strong className="text-foreground">Vast sequence space:</strong> 10^20+ possible CDR3 sequences
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>
                <strong className="text-foreground">Limited training data:</strong> Experimental validation is expensive
                and time-consuming
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>
                <strong className="text-foreground">Complex relationships:</strong> Structure-function relationships are
                non-linear
              </span>
            </li>
          </ul>
        </CardContent>
      </Card>

      {/* How ABYSS Solves It */}
      <Card>
        <CardHeader>
          <CardTitle>How ABYSS Solves This</CardTitle>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <Sparkles className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h4 className="font-semibold">ESM-2 Embeddings</h4>
                <p className="text-sm text-muted-foreground">
                  Leverage state-of-the-art protein language models to capture sequence semantics
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <Zap className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h4 className="font-semibold">Iterative Beam Search</h4>
                <p className="text-sm text-muted-foreground">
                  Efficiently explore sequence space with guided optimization
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <Database className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h4 className="font-semibold">Learned Scoring</h4>
                <p className="text-sm text-muted-foreground">
                  Train custom models on your experimental data for improved predictions
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <Dna className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <h4 className="font-semibold">Diversity Clustering</h4>
                <p className="text-sm text-muted-foreground">
                  Generate diverse candidate panels for experimental validation
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Getting Started */}
      <Card className="border-primary/50">
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>Follow these steps to design your CDR3 sequences</CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4">
          <Button
            variant="outline"
            className="h-auto flex-col items-start p-4 gap-2 bg-transparent"
            onClick={() => onNavigate("config")}
          >
            <div className="flex items-center gap-2 w-full">
              <Settings className="h-5 w-5" />
              <span className="font-semibold">1. Configure Parameters</span>
            </div>
            <p className="text-sm text-muted-foreground text-left">
              Set your antigen sequence and optimization parameters
            </p>
            <ArrowRight className="h-4 w-4 ml-auto" />
          </Button>

          <Button
            variant="outline"
            className="h-auto flex-col items-start p-4 gap-2 bg-transparent"
            onClick={() => onNavigate("optimize")}
          >
            <div className="flex items-center gap-2 w-full">
              <Zap className="h-5 w-5" />
              <span className="font-semibold">2. Run Optimization</span>
            </div>
            <p className="text-sm text-muted-foreground text-left">Generate and filter CDR3 candidates</p>
            <ArrowRight className="h-4 w-4 ml-auto" />
          </Button>

          <Button
            variant="outline"
            className="h-auto flex-col items-start p-4 gap-2 bg-transparent"
            onClick={() => onNavigate("results")}
          >
            <div className="flex items-center gap-2 w-full">
              <FileText className="h-5 w-5" />
              <span className="font-semibold">3. Review Results</span>
            </div>
            <p className="text-sm text-muted-foreground text-left">Analyze and download your candidate sequences</p>
            <ArrowRight className="h-4 w-4 ml-auto" />
          </Button>

          <Button
            variant="outline"
            className="h-auto flex-col items-start p-4 gap-2 bg-transparent"
            onClick={() => onNavigate("train")}
          >
            <div className="flex items-center gap-2 w-full">
              <GraduationCap className="h-5 w-5" />
              <span className="font-semibold">4. Train Custom Model (Optional)</span>
            </div>
            <p className="text-sm text-muted-foreground text-left">Improve predictions with your experimental data</p>
            <ArrowRight className="h-4 w-4 ml-auto" />
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
