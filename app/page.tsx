"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Settings, Zap, FileText, GraduationCap, BarChart3 } from "lucide-react"
import { OverviewPanel } from "@/components/overview-panel"
import { ConfigPanel } from "@/components/config-panel"
import { OptimizePanel } from "@/components/optimize-panel"
import { ResultsPanel } from "@/components/results-panel"
import { TrainPanel } from "@/components/train-panel"
import { EvaluatePanel } from "@/components/evaluate-panel"

export default function App() {
  const [activeTab, setActiveTab] = useState("overview")

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col items-center gap-2">
            <h1 className="text-3xl md:text-4xl font-bold text-center">ABYSS</h1>
            <p className="text-sm md:text-base text-muted-foreground text-center">
              Antibody Binding Yield through Sequence Searching
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 md:grid-cols-6 gap-2 h-auto p-2 bg-muted/50 mx-auto max-w-4xl">
            <TabsTrigger
              value="overview"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              {/* Placeholder for Home icon */}
              <span className="text-xs">Home</span>
            </TabsTrigger>
            <TabsTrigger
              value="config"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Settings className="h-5 w-5" />
              <span className="text-xs">Config</span>
            </TabsTrigger>
            <TabsTrigger
              value="optimize"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Zap className="h-5 w-5" />
              <span className="text-xs">Optimize</span>
            </TabsTrigger>
            <TabsTrigger
              value="results"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <FileText className="h-5 w-5" />
              <span className="text-xs">Results</span>
            </TabsTrigger>
            <TabsTrigger
              value="train"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <GraduationCap className="h-5 w-5" />
              <span className="text-xs">Train</span>
            </TabsTrigger>
            <TabsTrigger
              value="evaluate"
              className="flex flex-col items-center gap-1 py-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <BarChart3 className="h-5 w-5" />
              <span className="text-xs">Evaluate</span>
            </TabsTrigger>
          </TabsList>

          <div className="mt-6">
            <TabsContent value="overview" className="mt-0">
              <OverviewPanel onNavigate={setActiveTab} />
            </TabsContent>
            <TabsContent value="config" className="mt-0">
              <ConfigPanel />
            </TabsContent>
            <TabsContent value="optimize" className="mt-0">
              <OptimizePanel />
            </TabsContent>
            <TabsContent value="results" className="mt-0">
              <ResultsPanel />
            </TabsContent>
            <TabsContent value="train" className="mt-0">
              <TrainPanel />
            </TabsContent>
            <TabsContent value="evaluate" className="mt-0">
              <EvaluatePanel />
            </TabsContent>
          </div>
        </Tabs>
      </main>
    </div>
  )
}
