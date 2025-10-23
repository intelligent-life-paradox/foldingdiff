# Pipeline de Geração e Validação de Estruturas de Proteínas

Este documento descreve um fluxo de trabalho computacional para gerar novas estruturas de proteínas (backbones), projetar sequências de aminoácidos para essas estruturas e, em seguida, prever a estrutura 3D das sequências projetadas para validação.
A pipeline utiliza três ferramentas principais:

FoldingDiff: Para gerar novos backbones de proteínas.

ProteinMPNN: Para projetar sequências de aminoácidos que se encaixem nos backbones gerados.

OmegaFold: Para prever a estrutura 3D a partir das sequências de aminoácidos projetadas.
Pré-requisitos
Gerenciador de pacotes conda ou mamba.
GPU compatível com CUDA (os comandos foram executados com cudatoolkit=11.3).
Git para clonar os repositórios.

*Atualização: Proteínas geradas por OmegaFold na pasta proteinmpnn_residues (vale notar que nem todas as proteínas geradas pelo ProteinMPNN foram validadas por motivos de tempo e por ser somente uma demonstração)*



Passo 1: Geração de Backbones com FoldingDiff
Nesta etapa, usamos o FoldingDiff para gerar uma variedade de backbones de proteínas com comprimentos entre 50 e 128 resíduos.

1. Ative o ambiente Conda do FoldingDiff:
(Presume-se que você já tenha um ambiente chamado foldingdiff configurado).

code

*Bash*
conda activate foldingdiff

2. Navegue até o diretório do projeto e execute a amostragem:
Este comando irá gerar as estruturas no diretório generated_samples.

code

*Bash*
cd path/to/your/foldingdiff
python bin/sample.py \
    --l 50 128 \
    --n 1 \
    --b 8 \
    --device cuda:0 \ --outdir generated_samples #essas configurações de n e b valem para a minha máquina.Em projetos posteriores, com o auxílio da cloud, pode-se aumentar isso


--l 50 128: Gera proteínas com comprimentos que variam de 50 a 128 aminoácidos.
--n 1: Gera 1 exemplo para cada comprimento.
--b 8: Utiliza um tamanho de lote (batch size) de 8 para acelerar o processo.
--outdir generated_samples: Salva os arquivos .pdb de saída no diretório generated_samples.



Passo 2: Design de Sequências com ProteinMPNN
Com os backbones gerados, usamos o ProteinMPNN para projetar sequências de aminoácidos que provavelmente se dobrarão nessas estruturas.
1. Clone o repositório do ProteinMPNN:

code

*Bash*
git clone https://github.com/dauparas/ProteinMPNN.git

2. Crie e configure um novo ambiente Conda:
É uma boa prática usar um ambiente separado para evitar conflitos de dependência.
code

*Bash*
# Cria um ambiente com Python 3.9
conda create -n mlfold python=3.9

# Ativa o novo ambiente
conda activate mlfold

# Instala as dependências necessárias (PyTorch com suporte a CUDA)

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # essa versão do cudatoolkit deu certo para minha máquina. Provavelmente também dará para uma mais potente. 

3. Execute o ProteinMPNN para cada backbone gerado:
Este script itera sobre todos os arquivos .pdb da etapa anterior e gera 8 sequências candidatas para cada um.

code

*Bash*
# Navegue até o diretório do ProteinMPNN
cd ProteinMPNN

# Crie o diretório de saída para as sequências
mkdir -p ../proteinmpnn_residues/

# Loop para processar cada arquivo .pdb
for pdb_file in ../generated_samples/sampled_pdb/*.pdb; do
    echo "Processando $pdb_file..."
    python protein_mpnn_run.py \
            --pdb_path "$pdb_file" \
            --out_folder ../proteinmpnn_residues/ \
            --num_seq_per_target 8 \
            --sampling_temp "0.1" \
            --batch_size 1 # Processa um arquivo de cada vez
done
echo "Processamento do ProteinMPNN concluído!"


Os resultados serão salvos como arquivos .fasta no diretório proteinmpnn_residues/seqs/.
Passo 3: Predição de Estrutura com OmegaFold
Esta é a etapa de validação. Usamos o OmegaFold para prever a estrutura 3D das sequências geradas pelo ProteinMPNN. Isso nos ajuda a verificar se as sequências projetadas realmente se dobram na estrutura desejada.

1. Clone o repositório do OmegaFold:

code


*Bash*
git clone https://github.com/HeliXonProtein/OmegaFold.git
2. Crie e configure o ambiente Conda para o OmegaFold:


# Guia de Execução: Predição de Estrutura de Proteínas com OmegaFold no Lightning AI Studio



Observações importantes: Validação por OmegaFold foi feita via plataforma lightning.ai. Para o arquivo setup.py rodar é necessário criar um ambiente a parte.
Veja as instruções a seguir: 

#Guia de Execução: Predição de Estrutura de Proteínas com OmegaFold no Lightning AI Studio
Este guia descreve o processo completo para validar uma pasta de sequências de proteínas (.fa) usando o OmegaFold em um ambiente do Lightning AI.

Requisito Essencial: Python 3.9.18
 Atenção: O OmegaFold e suas dependências, especialmente as versões mais antigas do PyTorch compatíveis com certas GPUs, foram desenvolvidos e testados com a versão Python 3.9.18. O uso de outras versões (como 3.10, 3.11, etc.) pode resultar em erros de instalação de pacotes e problemas de incompatibilidade que impedirão a execução do programa. Esse foi um problema frequente, então é bom se atentar a isso.

Todo o processo descrito abaixo assume que o seu ambiente virtual (venv_omega) foi criado usando Python 3.9.18.
Passo a Passo

Siga estes passos diretamente no terminal do seu Lightning AI Studio.

Passo 1: Configuração e Verificação do Ambiente
É crucial ativar o ambiente virtual (venv) correto e verificar se ele está usando a versão do Python necessária.

Garanta que você está no diretório raiz. Se seu terminal mostra ~/OmegaFold, volte um nível:


code

*Bash*

cd ..

Ative o ambiente virtual:

code


*Bash*
source venv_omega/bin/activate

Verificação 1: Seu prompt do terminal deve agora começar com (venv_omega).
Verifique a versão do Python:

code

*Bash*
python --version

 Verificação 2: A saída deste comando deve ser Python 3.9.18. Se for diferente, o ambiente precisa ser recriado com a versão correta antes de prosseguir.

Passo 2: Instalação das Dependências

Se for a primeira vez executando ou após uma mudança de máquina, reinstale as dependências.
Navegue para a pasta do projeto:

code

*Bash*
cd OmegaFold

Instale os pacotes a partir do requirements.txt:

code


*Bash*

pip install -r requirements.txt

Passo 3: Preparação dos Dados de Entrada
As sequências precisam ser descompactadas e combinadas em um único arquivo para que o OmegaFold possa processá-las de forma eficiente.
Crie os diretórios de trabalho (se ainda não existirem):

code


*Bash*
# Pasta para as sequências descompactadas
mkdir -p sequencias_fasta

# Pasta para os resultados .pdb
mkdir -p resultados_pdb

Descompacte suas sequências. (O comando assume que seqs.zip está no diretório anterior ../).

code


*Bash*
unzip -o ../seqs.zip -d sequencias_fasta/
Combine todas as sequências em um único arquivo. (Ajuste o caminho se seus arquivos .fa estiverem em uma subpasta, como seqs/).

code


*Bash*
cat sequencias_fasta/seqs/*.fa > input_completo.fasta

 Verificação: Um novo arquivo chamado input_completo.fasta deve aparecer no seu diretório OmegaFold.
Passo 4: Execução do OmegaFold
Com o ambiente ativo e os dados preparados, inicie a predição.

Execute o comando principal do OmegaFold:

code


*Bash*
omegafold input_completo.fasta resultados_pdb/
