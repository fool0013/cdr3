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

export default function Page() {
  const [activeTab, setActiveTab] = useState("overview")

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col items-center gap-2">
            <h1 className="text-2xl sm:text-3xl font-bold text-foreground text-center">ABYSS</h1>
            <p className="text-xs sm:text-sm text-muted-foreground text-center">
              Antibody Binding Yield through Sequence Searching
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          {/* Navigation */}
          <TabsList className="grid w-full grid-cols-3 sm:grid-cols-6 gap-2 h-auto p-2 mb-6 bg-card/50">
            <TabsTrigger
              value="overview"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger
              value="config"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Settings className="h-4 w-4" />
              <span className="hidden sm:inline">Config</span>
            </TabsTrigger>
            <TabsTrigger
              value="optimize"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Zap className="h-4 w-4" />
              <span className="hidden sm:inline">Optimize</span>
            </TabsTrigger>
            <TabsTrigger
              value="results"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <FileText className="h-4 w-4" />
              <span className="hidden sm:inline">Results</span>
            </TabsTrigger>
            <TabsTrigger
              value="train"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <GraduationCap className="h-4 w-4" />
              <span className="hidden sm:inline">Train</span>
            </TabsTrigger>
            <TabsTrigger
              value="evaluate"
              className="flex items-center gap-2 min-h-[44px] data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <BarChart3 className="h-4 w-4" />
              <span className="hidden sm:inline">Evaluate</span>
            </TabsTrigger>
          </TabsList>

          {/* Tab Content */}
          <TabsContent value="overview" className="mt-0">
            <OverviewPanel />
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
        </Tabs>
      </main>
    </div>
  )
}
