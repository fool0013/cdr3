import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, Dna, Sparkles, Target } from "lucide-react"

export function OverviewPanel() {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="text-3xl">Welcome to ABYSS</CardTitle>
          <CardDescription className="text-base">
            Antibody Binding Yield through Sequence Searching - De novo antigen-specific CDR3 design
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground leading-relaxed">
            ABYSS is a machine learning-powered platform for designing novel CDR3 sequences with high binding affinity
            to target antigens. Using ESM-2 embeddings and iterative optimization, ABYSS explores the vast antibody
            sequence space to discover candidates with improved binding characteristics.
          </p>
        </CardContent>
      </Card>

      {/* Scientific Context */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Dna className="h-5 w-5" />
              What is CDR3?
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              The Complementarity Determining Region 3 (CDR3) is the most variable region of antibodies and the primary
              determinant of antigen specificity. Located at the center of the antigen-binding site, CDR3 sequences
              directly contact target antigens.
            </p>
            <p>
              CDR3 diversity arises from V(D)J recombination and somatic hypermutation, creating a vast sequence space
              (estimated at 10^15 possible sequences). This diversity enables the immune system to recognize virtually
              any antigen.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              The Challenge
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              Discovering high-affinity antibodies is challenging due to the enormous sequence space, limited training
              data from experimental screens, and complex structure-function relationships that are difficult to predict
              computationally.
            </p>
            <p>
              Traditional methods like phage display are time-consuming and expensive. Computational approaches can
              accelerate discovery by intelligently navigating sequence space to identify promising candidates for
              experimental validation.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* How ABYSS Solves This */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            How ABYSS Works
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            <strong className="text-foreground">ESM-2 Embeddings:</strong> Uses protein language models to capture
            sequence patterns and structural information without explicit 3D modeling.
          </p>
          <p>
            <strong className="text-foreground">Iterative Beam Search:</strong> Explores sequence space systematically
            by generating and scoring candidate mutations, keeping top performers for further optimization.
          </p>
          <p>
            <strong className="text-foreground">Learned Scoring:</strong> Trains on experimental binding data to predict
            which sequences will have high affinity for target antigens.
          </p>
          <p>
            <strong className="text-foreground">Diversity Clustering:</strong> Groups similar candidates to ensure
            diverse coverage of sequence space and avoid redundant designs.
          </p>
        </CardContent>
      </Card>

      {/* Getting Started */}
      <Card className="border-blue-500/50">
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>Follow these steps to design your CDR3 sequences</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="flex flex-col gap-2 p-4 rounded-lg bg-card border">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
                  1
                </span>
                Configure
              </div>
              <p className="text-xs text-muted-foreground">Set your antigen sequence and optimization parameters</p>
              <ArrowRight className="h-4 w-4 text-muted-foreground mt-auto" />
            </div>

            <div className="flex flex-col gap-2 p-4 rounded-lg bg-card border">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
                  2
                </span>
                Optimize
              </div>
              <p className="text-xs text-muted-foreground">Run the pipeline to generate and filter candidates</p>
              <ArrowRight className="h-4 w-4 text-muted-foreground mt-auto" />
            </div>

            <div className="flex flex-col gap-2 p-4 rounded-lg bg-card border">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
                  3
                </span>
                Review Results
              </div>
              <p className="text-xs text-muted-foreground">Analyze top candidates and download sequences</p>
              <ArrowRight className="h-4 w-4 text-muted-foreground mt-auto" />
            </div>

            <div className="flex flex-col gap-2 p-4 rounded-lg bg-card border">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs">
                  4
                </span>
                Validate
              </div>
              <p className="text-xs text-muted-foreground">Test experimentally and refine with training data</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
