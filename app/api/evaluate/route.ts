import { NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import { promises as fs } from "fs"
import path from "path"

const execAsync = promisify(exec)

export async function POST() {
  try {
    const configData = await fs.readFile(path.join(process.cwd(), "ui_state.json"), "utf-8")
    const config = JSON.parse(configData)

    const results: any = {}

    // Get global AUC
    try {
      const { stdout } = await execAsync("python eval_auc.py")
      const match = stdout.match(/AUC[:\s]+([0-9.]+)/)
      if (match) results.auc = Number.parseFloat(match[1])
    } catch (error) {
      console.log("[v0] Could not get AUC:", error)
    }

    // Get seed vs panel comparison
    if (config.antigen && config.seed_cdr3 && config.last_panel) {
      try {
        const { stdout } = await execAsync(
          `python seed_vs_panel.py "${config.antigen}" "${config.seed_cdr3}" "${config.last_panel}"`,
        )

        const seedMatch = stdout.match(/Seed.*?([0-9.]+)/)
        const meanMatch = stdout.match(/Panel mean.*?([0-9.]+)/)
        const maxMatch = stdout.match(/Panel max.*?([0-9.]+)/)

        if (seedMatch) results.seed_score = Number.parseFloat(seedMatch[1])
        if (meanMatch) results.panel_mean = Number.parseFloat(meanMatch[1])
        if (maxMatch) results.panel_max = Number.parseFloat(maxMatch[1])

        if (results.seed_score && results.panel_max) {
          results.improvement = (results.panel_max - results.seed_score) / results.seed_score
        }
      } catch (error) {
        console.log("[v0] Could not compare seed vs panel:", error)
      }
    }

    return NextResponse.json(results)
  } catch (error) {
    console.error("[v0] Evaluate error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Evaluation failed" }, { status: 500 })
  }
}
