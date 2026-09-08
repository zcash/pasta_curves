use clap::{Parser, Subcommand};

mod asm_mutants;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    AsmMutants {
        /// Check the tracked survivor list against the assembly without
        /// running any mutants.
        #[arg(long)]
        verify_list: bool,

        /// Print the mutation set as `id`, line, instruction and replacement,
        /// tab separated, so other tools need not re-derive it.
        #[arg(long)]
        list_sites: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::AsmMutants {
            verify_list,
            list_sites,
        } => asm_mutants::run(verify_list, list_sites),
    }
}
