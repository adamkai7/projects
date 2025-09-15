import streamlit as st
import random
import math
from calculus import generate_calculus
from trignometry import generate_trigonometry
from arithmetic import generate_basic_arithmetic 
from fractions2 import generate_fractions
from algebra import generate_algebra
from geometry import generate_geometry
from word_problems import generate_word_problems
from utils import create_download_buttons
from truthtable import generate_truthtable


st.title("Number Ninja")
st.write("Generate custom math problems based on your preferences!")

# Sidebar for concept selection
st.sidebar.header("Select Math Concepts")
selected_concepts = st.sidebar.multiselect(
    "Choose concepts you want to practice:",
    ["Calculus", "Trignometry", "Basic Arithmetic", "Fractions", "Algebra", "Geometry", "Word Problems", "Truth Table"],
    default=["Calculus"]
)

if not selected_concepts:
    st.info("Please select at least one math concept from the sidebar to generate problems.")
else:
    concept_generators = {
        "Calculus": generate_calculus,
        "Trignometry": generate_trigonometry,
        "Basic Arithmetic": generate_basic_arithmetic,
        "Fractions": generate_fractions,
        "Algebra": generate_algebra,
        "Geometry": generate_geometry,
        "Word Problems": generate_word_problems,
        "Truth Table": generate_truthtable
    }
    
    st.write("## Problem Settings")

    # Use session state to persist values
    if "num_problems" not in st.session_state:
        st.session_state.num_problems = 5

    st.session_state.num_problems = st.number_input(
        "Number of problems to generate per concept:",
        min_value=1, max_value=20, value=st.session_state.num_problems,
        key="num_problems_input"
    )

    # Force a re-generation if options change
    if "regen_key" not in st.session_state:
        st.session_state.regen_key = 0

    def trigger_regeneration():
        st.session_state.regen_key += 1

    all_problems = {}
    all_solutions = {}
    all_word_problems = {}

    for concept in selected_concepts:
        st.write(f"### {concept}")

        # Handle calculus separately since it returns 3 values
        if concept == "Calculus":
            problems, solutions, word_problems = concept_generators[concept](
                concept, st.session_state.num_problems
            )
            all_word_problems[concept] = word_problems
        else:
            # Other generators return only problems and solutions
            problems, solutions = concept_generators[concept](
                concept, st.session_state.num_problems
            )

        all_problems[concept] = problems
        all_solutions[concept] = solutions

        # Display problems with word problems for Calculus
        for i, problem in enumerate(problems):
            if concept == "Calculus":
                st.write(f"**Word Problem {i+1}:** {all_word_problems[concept][i]}")
                st.write(f"**Mathematical Formulation {i+1}:** {problem}")
            else:
                st.write(f"**Problem {i+1}:** {problem}")

    with st.expander("View Solutions"):
        for concept in selected_concepts:
            st.write(f"### {concept} Solutions")
            for i, solution in enumerate(all_solutions[concept]):
                st.write(f"**Problem {i+1}:** {solution}")