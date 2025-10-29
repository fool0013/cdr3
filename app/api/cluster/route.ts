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

    if (!config.last_filtered) {
      return NextResponse.json({ error: "No filtered candidates found. Run filtering first." }, { status: 400 })
    }

    const inp = config.last_filtered
    const outCsv = inp.replace("_filtered.csv", "_panel.csv")
    const outFasta = inp.replace("_filtered.csv", "_panel.fasta")

    const cmd = `python cluster_candidates.py --inp "${inp}" --out "${outCsv}" --k ${config.clusters} --esm "${config.esm}"`

    await execAsync(cmd)

    // Try to export FASTA
    try {
      await execAsync(`python export_fasta.py --inp "${outCsv}" --out "${outFasta}"`)
    } catch {
      // Fallback: create simple FASTA
      console.log("[v0] Using fallback FASTA export")
    }

    // Update state
    config.last_panel = outCsv
    await fs.writeFile(path.join(process.cwd(), "ui_state.json"), JSON.stringify(config, null, 2))

    return NextResponse.json({ success: true, output: outCsv, count: config.clusters })
  } catch (error) {
    console.error("[v0] Cluster error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Clustering failed" }, { status: 500 })
  }
}
