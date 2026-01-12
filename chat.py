import torch
import torch.nn.functional as F
import sys
import os
import psutil  # <--- NEW: For memory tracking
from colorama import Fore, Style, init

# Import from your training script
from train_hope import HOPE, CONFIG, DEVICE

# Initialize colors
init(autoreset=True)

def get_memory_usage():
    """Returns the RAM usage of the current Python process in MB."""
    process = psutil.Process(os.getpid())
    mb = process.memory_info().rss / 1024 / 1024
    return mb

def print_model_stats(model):
    """Calculates and prints model size and memory footprint."""
    
    # 1. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Measure Memory
    # Force a garbage collection/cache clear to get accurate reading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    current_ram = get_memory_usage()
    
    print(f"\n{Fore.GREEN}=== MODEL STATISTICS ==={Style.RESET_ALL}")
    print(f"Architecture:   HOPE (Nested Learning)")
    print(f"Parameters:     {Fore.YELLOW}{total_params / 1_000_000:.2f} Million{Style.RESET_ALL}")
    
    # Estimate size on disk (Float32 = 4 bytes per param)
    size_on_disk = (total_params * 4) / 1024 / 1024
    print(f"Est. File Size: {size_on_disk:.1f} MB")
    
    # Actual RAM usage of the script
    print(f"Active Memory:  {Fore.CYAN}{current_ram:.1f} MB{Style.RESET_ALL} (Unified/System RAM)")
    
    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU VRAM:       {vram:.1f} MB")
        
    print("="*30 + "\n")

def load_model(path):
    print(f"{Fore.YELLOW}Loading model from {path}...{Style.RESET_ALL}")
    try:
        # 1. Create the empty architecture
        model = HOPE(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_layers'])
        
        # 2. Load the file from disk
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # 3. INTELLIGENT UNPACKING
        # If we saved extra data (like step count), extract just the weights
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            print(f"{Fore.CYAN}Detected Smart Checkpoint (includes training step). Unpacking...{Style.RESET_ALL}")
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint # It's just the weights (Legacy format)

        # 4. Inject weights into model
        model.load_state_dict(state_dict, strict=False) 
        
        model.to(DEVICE)
        model.eval()
        return model
        
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Model file '{path}' not found.{Style.RESET_ALL}")
        print(f"Current Config points to: {CONFIG['save_path']}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{Fore.RED}Architecture Mismatch!{Style.RESET_ALL}")
        print("Your saved model has different dimensions than your current CONFIG.")
        print(f"Error details: {e}")
        sys.exit(1)

def generate_response(model, prompt, max_new_tokens=250, temperature=0.3):
    # --- TEMPLATE FIX: Match the fine-tuning format ---
    # Model was trained with "Question: ... \nAnswer: ... \nReasoning: ..."
    full_prompt = f"Question: {prompt.strip()}\nAnswer: "
    
    input_ids = list(full_prompt.encode('utf-8'))
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print(f"\n{Fore.CYAN}HOPE: {Style.RESET_ALL}", end="")
    
    generated_bytes = []
    byte_buffer = bytearray()
    
    with torch.no_grad():
        logits, state = model(x)
    
    last_token_logits = logits[:, -1, :] / temperature
    probs = F.softmax(last_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    for _ in range(max_new_tokens):
        token_int = next_token.item()
        
        # Stop on 0 (This is our padding/stop token from isolated training)
        if token_int == 0: 
            break 
        
        generated_bytes.append(token_int)
        
        # --- SMART DECODING & PRINTING ---
        if token_int == 32: # Space
            byte_buffer.clear() 
            sys.stdout.write(" ")
        elif token_int == 10: # Newline
            byte_buffer.clear()
            sys.stdout.write("\n")
        else:
            byte_buffer.append(int(token_int))
            try:
                decoded_char = byte_buffer.decode('utf-8')
                sys.stdout.write(decoded_char)
                byte_buffer.clear()
            except UnicodeDecodeError:
                pass
        
        sys.stdout.flush()
        
        with torch.no_grad():
            x = next_token 
            logits, state = model(x, state=state)
            
        last_token_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    print() # Final newline
    return bytes(generated_bytes).decode('utf-8', errors='ignore')

def main():
    # Use the English model path
    model_path = CONFIG['save_path']  # Will use hope_en_deep.pth from CONFIG
    
    # 1. Load
    model = load_model(model_path)
    
    # 2. Show Stats (The new part)
    print_model_stats(model)
    
    print("Interactive Console - Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n{Fore.WHITE}You: {Style.RESET_ALL}")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if not user_input.strip():
                continue
                
            generate_response(model, user_input)
            
            # Optional: Show memory spike after generation
            # print(f"{Fore.BLACK}{Style.DIM}[Mem: {get_memory_usage():.0f}MB]{Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
