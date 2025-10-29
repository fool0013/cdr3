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

    if (!config.antigen) {
      return NextResponse.json({ error: "No antigen sequence configured" }, { status: 400 })
    }

    const outDir = config.out_folder || "runs"
    await fs.mkdir(outDir, { recursive: true })

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)
    const outCsv = path.join(outDir, `opt_${timestamp}.csv`)

    const cmd = `python optimize_cdr3.py --antigen "${config.antigen}" --start_cdr3 "${config.seed_cdr3}" --steps ${config.steps} --beam ${config.beam} --topk 20 --k_mut ${config.k_mut} --ckpt "${config.checkpoint}" --out_csv "${outCsv}"`

    await execAsync(cmd)

    // Update state with last output
    config.last_raw = outCsv
    await fs.writeFile(path.join(process.cwd(), "ui_state.json"), JSON.stringify(config, null, 2))

    return NextResponse.json({ success: true, output: outCsv, count: config.beam })
  } catch (error) {
    console.error("[v0] Optimize error:", error)
    return NextResponse.json({ error: error instanceof Error ? error.message : "Optimization failed" }, { status: 500 })
  }
}
