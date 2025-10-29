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

    if (!config.last_raw) {
      return NextResponse.json({ error: "No raw candidates found. Run optimization first." }, { status: 400 })
    }

    const inp = config.last_raw
    const outCsv = inp.replace(".csv", "_filtered.csv")

    const cmd = `python filter_cdr3s.py --inp "${inp}" --out "${outCsv}" --keep_top ${config.keep_top}`

    await execAsync(cmd)

    // Update state
    config.last_filtered = outCsv
    await fs.writeFile(path.join(process.cwd(), "ui_state.json"), JSON.stringify(config, null, 2))

    return NextResponse.json({ success: true, output: outCsv, count: config.keep_top })
  } catch (error) {
    console.error("[v0] Filter error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Filtering failed" }, { status: 500 })
  }
}
