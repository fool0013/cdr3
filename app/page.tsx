"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Home, Settings, Zap, FileText, GraduationCap, BarChart3 } from "lucide-react"
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
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col items-center gap-2 text-center">
            <h1 className="text-3xl sm:text-4xl font-bold text-foreground">ABYSS</h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              Antibody Binding Yield through Sequence Searching
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 max-w-7xl">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="flex justify-center mb-8">
            <TabsList className="inline-flex h-auto p-1 bg-muted/50 backdrop-blur-sm">
              <TabsTrigger
                value="overview"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <Home className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Home</span>
              </TabsTrigger>
              <TabsTrigger
                value="config"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <Settings className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Config</span>
              </TabsTrigger>
              <TabsTrigger
                value="optimize"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <Zap className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Optimize</span>
              </TabsTrigger>
              <TabsTrigger
                value="results"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <FileText className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Results</span>
              </TabsTrigger>
              <TabsTrigger
                value="train"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <GraduationCap className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Train</span>
              </TabsTrigger>
              <TabsTrigger
                value="evaluate"
                className="flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2 min-h-[44px] data-[state=active]:bg-background"
              >
                <BarChart3 className="h-4 w-4" />
                <span className="text-xs sm:text-sm">Evaluate</span>
              </TabsTrigger>
            </TabsList>
          </div>

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
