use clap::{Parser, Subcommand};

mod asm_mutants;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    AsmMutants,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::AsmMutants => asm_mutants::run(),
    }
}
