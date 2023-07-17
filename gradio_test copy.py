import os
import sys

import fire
import gradio as gr

def main():

    def check_user():
        return True

    demo = gr.Blocks(title="Raven", analytics_enabled=True)
    callback = gr.CSVLogger()

    with demo:    
        gr.Markdown(
        """
            <p style="text-align:center; font-size:3em;">            
                📈 Raven 📈        
            </p>
        """
        )
        gr.Markdown(
            """
            <p style="text-align: center;">
                Raven is a 7B-parameter LLaMA model finetuned to follow instructions in the finance domain. <br/>
                It is trained on the <a href="https://github.com/tatsu-lab/stanford_alpaca">Stanford Alpaca</a> dataset and several other finance datasets. For more information, please visit <a href="https://github.com/adriantheuma/fin-expert">the project's website</a>.  
            </p>
            """
        )
        with gr.Row():   
            with gr.Column(scale=50):     
                with gr.Row(variant="panel"):                     
                    instruction = gr.components.Textbox(
                        lines=2,
                        label="Instruction",                
                        placeholder="How is a raven related to wealth?",
                    )
                with gr.Row(variant="panel"):     
                    input = gr.components.Textbox(
                        lines=2, 
                        label="Input", 
                        info="Provide further context such as a passage.",
                        placeholder="none"
                    )   
                with gr.Row(variant="panel"):    
                    temperature = gr.components.Slider(
                        minimum=0, maximum=1, value=0.1, label="Temperature", info="A higher temperature value typically makes the output more diverse and creative but might also increase its likelihood of straying from the context."
                    )
                    top_k = gr.components.Slider(
                        minimum=0, maximum=100, step=1, value=10, label="Top-k", info="Top-k tells the model to pick the next token from the top ‘k’ tokens in its list, sorted by probability."
                    )
                    top_p = gr.components.Slider(
                        minimum=0, maximum=1, value=0.75, label="Top-p", info="Top-p picks from the top-k tokens based on the sum of their probabilities."
                    )   
                with gr.Row(variant="panel"):
                    beams = gr.components.Slider(
                        minimum=1, maximum=4, step=1, value=2, label="Beams", info="The final output is the most probable sequence of tokens found among the k beams."
                    )
                    max_tokens = gr.components.Slider(
                        minimum=1, maximum=2000, step=1, value=256, label="Max tokens", info="Stopping criteria for the model"
                    )
                    stream = gr.components.Checkbox(
                        label="Stream output", value=True, info="Stream the output one token at a time."
                    )   
                with gr.Row():
                    inputs = [temperature, top_k, top_p, beams, max_tokens, stream]
                    clear_btn = gr.ClearButton(value="Clear")
                    submit_btn = gr.components.Button(value="Submit", variant="primary")
                    
            with gr.Column(scale=50, variant="panel"):
                output = gr.inputs.Textbox(
                    lines=10,
                    label="Output",
                )
                flag_btn = gr.Button("Flag this output", variant="stop")
        
        
        inputs = [instruction, input, temperature, top_k, top_p, beams, max_tokens, stream]
        outputs = [output]
        
        callback.setup(inputs + outputs, "flagged_data_points")

        components_to_clear = [instruction, input, output]
        clear_btn.add(components=components_to_clear)

        # submit_btn.click(fn=evaluate, inputs=inputs, outputs=outputs)

        # We can choose which components to flag -- in this case, we'll flag all of them
        flag_btn.click(
            fn=lambda *args: callback.flag(args), 
            inputs=inputs + outputs, 
            outputs=None, 
            preprocess=False,
            show_progress="full"
        )
        

    demo.queue().launch(
        server_name="0.0.0.0", 
        share=False,
        auth=("user", "HJ49AJnXy36kKYTg")
    )

if __name__ == "__main__":
    fire.Fire(main)