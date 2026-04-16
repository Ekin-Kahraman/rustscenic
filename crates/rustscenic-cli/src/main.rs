use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rustscenic")]
#[command(about = "Fast SCENIC+ stage replacements")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// GRN inference (GRNBoost2 replacement) — v0.1
    Grn {
        #[arg(long)]
        expression: String,
        #[arg(long)]
        tfs: String,
        #[arg(long)]
        output: String,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long, default_value_t = 777)]
        seed: u64,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Grn { .. } => {
            anyhow::bail!("v0.1 grn implementation pending");
        }
    }
}
