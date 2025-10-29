import { NextResponse } from "next/server"
import { promises as fs } from "fs"
import path from "path"

export async function GET() {
  try {
    const configData = await fs.readFile(path.join(process.cwd(), "ui_state.json"), "utf-8")
    const config = JSON.parse(configData)

    const results: any = {
      raw: config.last_raw || null,
      filtered: config.last_filtered || null,
      panel: config.last_panel || null,
      candidates: [],
    }

    // Try to load panel candidates for preview
    if (config.last_panel) {
      try {
        const csvData = await fs.readFile(path.join(process.cwd(), config.last_panel), "utf-8")
        const lines = csvData.split("\n").filter((l) => l.trim())

        // Skip header, parse up to 20 rows
        for (let i = 1; i < Math.min(21, lines.length); i++) {
          const parts = lines[i].split(",")
          if (parts.length >= 2) {
            results.candidates.push({
              cdr3: parts[1].trim(),
              score: Number.parseFloat(parts[2]) || 0,
              cluster: parts[3] ? Number.parseInt(parts[3]) : undefined,
            })
          }
        }
      } catch (error) {
        console.log("[v0] Could not load candidates preview:", error)
      }
    }

    return NextResponse.json(results)
  } catch (error) {
    return NextResponse.json({ error: "Failed to load results" }, { status: 500 })
  }
}
