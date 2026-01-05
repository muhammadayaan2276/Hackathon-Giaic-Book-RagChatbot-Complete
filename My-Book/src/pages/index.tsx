import type { ReactNode } from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Head from "@docusaurus/Head";
import styles from "./index.module.css";

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout>
      <Head>
        <title>My Book</title>
        <meta
          name="description"
          content="Physical AI & Humanoid Robotics Book"
        />
      </Head>

      <header className={styles.hero}>
        <div className={styles.overlay}></div>

        <div className={styles.heroContent}>
          <h1 className={styles.title}>Spec Driven Book Creation With AI</h1>
          <p className={styles.subtitle}>Master AI-Powered Book Writing</p>

          <p className={styles.description}>
            A complete guide to building intelligent humanoid robots using
            ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.
          </p>

          <div className={styles.buttons}>
            <Link
              className={clsx("button button--secondary", styles.ctaBtn)}
              to="/docs/intro"
            >
              Start Learning 🚀
            </Link>

            <Link
              className={clsx("button button--primary", styles.secondaryBtn)}
              to="/docs/category/module-1-ros-2"
            >
              Explore Modules 📚
            </Link>


          </div>
        </div>
      </header>

      <main className={styles.mainSection}>
        <section className={styles.infoSection}>
          <h2>Build Intelligent Humanoid Robots</h2>
          <p>
            Learn the full stack of Physical AI—from robot middleware and simulation
            to advanced perception and LLM-driven planning. This book teaches you
            everything needed to design, simulate, and control humanoid robots.
          </p>
        </section>
      </main>
    </Layout>
  );
}
