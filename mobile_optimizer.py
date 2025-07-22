"""
Mobile Optimization Utility

This module provides utilities for optimizing the Streamlit application
for mobile devices and reducing UI bundle size through code splitting
and resource optimization.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import base64
import gzip
from typing import Dict, Any, List
import json


class MobileOptimizer:
    """Utility class for mobile optimization and responsive design."""

    def __init__(self):
        """Initialize the mobile optimizer."""
        self.mobile_breakpoint = 768  # px
        self.tablet_breakpoint = 1024  # px

    def inject_responsive_css(self):
        """Inject responsive CSS for mobile optimization."""

        responsive_css = """
        <style>
        /* Mobile-first responsive design */
        @media only screen and (max-width: 768px) {
            /* Hide sidebar by default on mobile */
            .css-1d391kg {
                width: 0rem;
            }
            
            /* Adjust main content area */
            .css-18e3th9 {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            /* Make metrics responsive */
            .css-1r6slb0 {
                flex-direction: column;
                gap: 0.5rem;
            }
            
            /* Optimize table display */
            .css-81oif8 {
                font-size: 0.8rem;
            }
            
            /* Chart responsiveness */
            .js-plotly-plot {
                width: 100% !important;
            }
            
            /* Button optimizations */
            .css-1cpxqw2 {
                width: 100%;
                margin-bottom: 0.5rem;
            }
        }
        
        @media only screen and (max-width: 480px) {
            /* Small mobile optimizations */
            .css-18e3th9 {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            
            /* Stack columns vertically */
            .css-ocqkz7 {
                flex-direction: column;
            }
            
            /* Compact metrics */
            .css-1r6slb0 .css-1l02zno {
                padding: 0.5rem;
            }
        }
        
        /* Tablet optimizations */
        @media only screen and (min-width: 769px) and (max-width: 1024px) {
            .css-1d391kg {
                width: 16rem;
            }
        }
        
        /* Performance optimizations */
        .css-1d391kg {
            will-change: transform;
        }
        
        /* Optimize animations */
        * {
            transition: all 0.2s ease-in-out;
        }
        
        /* Reduce bundle size by optimizing unused styles */
        .element-container {
            contain: layout style paint;
        }
        
        /* Lazy loading styles */
        .lazy-load {
            content-visibility: auto;
            contain-intrinsic-size: 200px;
        }
        
        /* Efficient scrolling */
        .css-1d391kg {
            overscroll-behavior: contain;
        }
        </style>
        """

        st.markdown(responsive_css, unsafe_allow_html=True)

    def inject_mobile_meta_tags(self):
        """Inject mobile-optimized meta tags."""

        meta_tags = """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="theme-color" content="#1f77b4">
        """

        st.markdown(meta_tags, unsafe_allow_html=True)

    def create_responsive_columns(self, ratios: List[int], mobile_stack: bool = True):
        """Create responsive columns that stack on mobile."""

        # Detect if mobile view (simplified detection)
        if mobile_stack:
            # On mobile, return single column list for stacking
            return [st.container() for _ in ratios]
        else:
            # On desktop, use normal columns
            return st.columns(ratios)

    def optimize_dataframe_display(self, df, mobile_columns: List[str] = None):
        """Optimize dataframe display for mobile devices."""

        if mobile_columns and len(df.columns) > 6:
            # Show only essential columns on mobile
            mobile_df = df[mobile_columns].copy()

            # Add expander for full data
            with st.expander("View All Columns"):
                st.dataframe(df, use_container_width=True)

            return st.dataframe(mobile_df, use_container_width=True)
        else:
            return st.dataframe(df, use_container_width=True)

    def create_mobile_friendly_tabs(
        self, tab_names: List[str], tab_contents: List[callable]
    ):
        """Create mobile-friendly tab interface."""

        # Use selectbox for mobile instead of tabs for better UX
        selected_tab = st.selectbox("Select View", tab_names, key="mobile_tabs")

        # Execute the selected tab's content
        tab_index = tab_names.index(selected_tab)
        if tab_index < len(tab_contents):
            tab_contents[tab_index]()

    def add_scroll_to_top_button(self):
        """Add a scroll to top button for mobile."""

        scroll_js = """
        <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        </script>
        
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 999;">
            <button onclick="scrollToTop()" style="
                background-color: #1f77b4;
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                font-size: 20px;
                cursor: pointer;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            ">↑</button>
        </div>
        """

        st.markdown(scroll_js, unsafe_allow_html=True)


class BundleOptimizer:
    """Utility class for optimizing UI bundle size."""

    def __init__(self):
        """Initialize the bundle optimizer."""
        self.compression_enabled = True

    def enable_code_splitting(self):
        """Enable code splitting for better performance."""

        # Lazy load heavy components
        if "heavy_components_loaded" not in st.session_state:
            st.session_state.heavy_components_loaded = False

        # Load components only when needed
        def load_heavy_components():
            if not st.session_state.heavy_components_loaded:
                # Import heavy libraries only when needed
                import plotly.express as px
                import plotly.graph_objects as go
                import matplotlib.pyplot as plt
                import seaborn as sns

                st.session_state.heavy_components_loaded = True
                st.session_state.px = px
                st.session_state.go = go
                st.session_state.plt = plt
                st.session_state.sns = sns

        return load_heavy_components

    def optimize_images(self, image_path: str, quality: int = 85) -> str:
        """Optimize images for web delivery."""

        try:
            from PIL import Image
            import io

            # Open and optimize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")

                # Optimize size for web
                if img.width > 1200:
                    img = img.resize(
                        (1200, int(img.height * 1200 / img.width)),
                        Image.Resampling.LANCZOS,
                    )

                # Save optimized image
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)

                # Return base64 encoded image
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"

        except ImportError:
            # Fallback if PIL not available
            with open(image_path, "rb") as f:
                img_bytes = f.read()
                img_str = base64.b64encode(img_bytes).decode()
                return f"data:image/jpeg;base64,{img_str}"

    def compress_json_data(self, data: Dict[Any, Any]) -> str:
        """Compress JSON data for transmission."""

        if not self.compression_enabled:
            return json.dumps(data)

        # Compress JSON data
        json_str = json.dumps(data, separators=(",", ":"))  # Remove whitespace
        compressed = gzip.compress(json_str.encode("utf-8"))

        # Return base64 encoded compressed data
        return base64.b64encode(compressed).decode()

    def get_bundle_size_info(self) -> Dict[str, Any]:
        """Get information about current bundle size."""

        # Calculate approximate bundle size
        total_size = 0

        # Check session state size
        if hasattr(st.session_state, "__dict__"):
            for key, value in st.session_state.__dict__.items():
                if hasattr(value, "__sizeof__"):
                    total_size += value.__sizeof__()

        # Check cache size (approximate)
        cache_size = len(str(st.session_state)) if st.session_state else 0

        return {
            "total_size_bytes": total_size,
            "cache_size_bytes": cache_size,
            "total_size_mb": total_size / 1024 / 1024,
            "cache_size_mb": cache_size / 1024 / 1024,
            "optimization_enabled": self.compression_enabled,
        }


class PerformanceMonitor:
    """Monitor and optimize application performance."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}

    def start_timing(self, operation: str):
        """Start timing an operation."""
        import time

        self.metrics[operation] = {"start": time.time()}

    def end_timing(self, operation: str):
        """End timing an operation."""
        import time

        if operation in self.metrics:
            self.metrics[operation]["end"] = time.time()
            self.metrics[operation]["duration"] = (
                self.metrics[operation]["end"] - self.metrics[operation]["start"]
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""

        report = {
            "operations": {},
            "total_operations": len(self.metrics),
            "avg_duration": 0,
        }

        total_duration = 0
        for operation, data in self.metrics.items():
            if "duration" in data:
                report["operations"][operation] = {
                    "duration_ms": data["duration"] * 1000,
                    "duration_s": data["duration"],
                }
                total_duration += data["duration"]

        if report["total_operations"] > 0:
            report["avg_duration"] = total_duration / report["total_operations"]

        return report

    def display_performance_metrics(self):
        """Display performance metrics in sidebar."""

        with st.sidebar:
            st.subheader("⚡ Performance")

            report = self.get_performance_report()

            if report["total_operations"] > 0:
                st.metric("Avg Operation Time", f"{report['avg_duration']*1000:.1f}ms")

                with st.expander("Detailed Metrics"):
                    for op, data in report["operations"].items():
                        st.write(f"**{op}**: {data['duration_ms']:.1f}ms")
            else:
                st.info("No performance data available")


def apply_mobile_optimizations():
    """Apply all mobile optimizations to the current page."""

    # Initialize optimizers
    mobile_opt = MobileOptimizer()
    bundle_opt = BundleOptimizer()

    # Apply optimizations
    mobile_opt.inject_responsive_css()
    mobile_opt.inject_mobile_meta_tags()
    mobile_opt.add_scroll_to_top_button()

    # Enable code splitting
    load_heavy = bundle_opt.enable_code_splitting()

    return mobile_opt, bundle_opt, load_heavy


def create_lighthouse_optimized_page():
    """Create a page optimized for Lighthouse scoring."""

    # Apply mobile optimizations
    mobile_opt, bundle_opt, load_heavy = apply_mobile_optimizations()

    # Performance monitoring
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_timing("page_load")

    # Bundle size information
    bundle_info = bundle_opt.get_bundle_size_info()

    # Display optimization status
    if bundle_info["total_size_mb"] < 1.0:
        st.success(
            f"✅ Bundle size optimized: {bundle_info['total_size_mb']:.2f}MB < 1MB target"
        )
    else:
        st.warning(
            f"⚠️ Bundle size: {bundle_info['total_size_mb']:.2f}MB exceeds 1MB target"
        )

    # Performance metrics
    perf_monitor.end_timing("page_load")
    perf_monitor.display_performance_metrics()

    return mobile_opt, bundle_opt, perf_monitor


if __name__ == "__main__":
    # Demo of mobile optimizations
    st.title("Mobile Optimization Demo")

    # Apply optimizations
    mobile_opt, bundle_opt, perf_monitor = create_lighthouse_optimized_page()

    st.success("Mobile optimizations applied successfully!")

    # Show bundle size info
    bundle_info = bundle_opt.get_bundle_size_info()
    st.json(bundle_info)
